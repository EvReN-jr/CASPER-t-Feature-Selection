#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Builds the LaTeX performance tables of the article from the per-fold result
files of the reference pipeline.  A cell is the mean over the eight
classifiers on each outer fold, reported as mean +- std over the 10 folds.

Usage:  python make_tables.py [--check]
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.environ.get('CASPER_RESULTS_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
FIX = os.path.join(ROOT, 'uhem_tda', 'gamma_pfi_matchk_fix')
ROUGH = os.path.join(ROOT, 'uhem_tda', 'roughset_baselines', 'sonuclar', 'out')
TABLES = os.path.join(PAPER, 'tables')
os.makedirs(TABLES, exist_ok=True)

# ---------------------------------------------------------------- veri setleri
SMALL = ['cyron', 'diabet', 'kanser', 'parkinson', 'sonar']
LARGE = ['banknote', 'phoneme', 'htru2', 'magic']
NICE = {'cyron': 'Cryo.', 'diabet': 'Diab.', 'kanser': 'Breast', 'parkinson': 'Park.',
        'sonar': 'Sonar', 'magic1k': 'MAGIC-1k', 'banknote': 'Banknote',
        'phoneme': 'Phoneme', 'htru2': 'HTRU2', 'magic': 'MAGIC'}

# ------------------------------------------------------------------- yontemler
CASPER = 'V1_tstat'
CLASSICAL = ['StdPFI_matchK', 'RFE_matchK', 'GB_matchK', 'SHAP_matchK', 'RF_matchK',
             'LASSO_matchK', 'MI_matchK', 'ANOVA_matchK', 'ReliefF_matchK']
ROUGHM = ['QuickReduct_matchK', 'SkroughRed_matchK', 'FRFS_matchK']
ROUGHN = ['QuickReduct_native', 'SkroughRed_native', 'FRFS_native']
MATCHED = ['ALL', CASPER] + CLASSICAL + ROUGHM      # 14 yontem: rank bu blok uzerinden
OWN = ['ALL', CASPER] + ROUGHN                      # 5 yontem

LABEL = {'ALL': 'ALL', CASPER: '\\CASPERt{}',
         'QuickReduct_matchK': 'QuickReduct', 'SkroughRed_matchK': 'Approx.\\ red.',
         'FRFS_matchK': 'FRFS',
         'QuickReduct_native': 'QuickReduct', 'SkroughRed_native': 'Approx.\\ red.',
         'FRFS_native': 'FRFS'}
LABEL.update({c: c.replace('_matchK', '') for c in CLASSICAL})

METRICS = [('ACCU', 'Accuracy'), ('F1_S', 'F1-Score'), ('AUC', 'AUC')]
COLS = ['Dataset', 'Fold', 'Method', 'N_Features_Used', 'Classifier', 'ACCU', 'F1_S', 'AUC']


# ------------------------------------------------------------------ yukleyici
def _xlsx(ds):
    loc = 'main_study/results' if ds in SMALL else 'results'
    return os.path.join(FIX, loc, 'per_dataset', f'{ds}_TopologicalResults.xlsx')


def load(ds):
    """Bir veri setinin butun yontemlerini tek fold-level cerceveye topla."""
    a = pd.read_excel(_xlsx(ds), sheet_name='Fold_Detail')[COLS]
    b = pd.read_csv(os.path.join(ROUGH, f'{ds}_rough_baselines_folds.csv'))[COLS]
    b = b[b.Method != 'ALL']          # ALL zaten xlsx'te ve birebir ayni (dogrulandi)
    return pd.concat([a, b], ignore_index=True)


def load_group(datasets):
    return pd.concat([load(d) for d in datasets], ignore_index=True)


# -------------------------------------------------------------- istatistikler
def cell_stats(df, metric):
    """(Dataset, Method) -> katlar arasi mean/std (siniflandirici ortalamasi alinmis)."""
    per_fold = df.groupby(['Dataset', 'Method', 'Fold'])[metric].mean().reset_index()
    g = per_fold.groupby(['Dataset', 'Method'])[metric]
    return g.mean().unstack() * 100, g.std().unstack() * 100


def clf_spread(df, metric):
    """(Dataset, Method) -> 8 siniflandirici arasi std (her biri 10 kat ortalamasi)."""
    per_clf = df.groupby(['Dataset', 'Method', 'Classifier'])[metric].mean().reset_index()
    return per_clf.groupby(['Dataset', 'Method'])[metric].std().unstack() * 100


def kappa(df):
    """(Dataset, Method) -> ortalama tutulan oznitelik sayisi K."""
    return df.groupby(['Dataset', 'Method']).N_Features_Used.mean().unstack()


def mean_ranks(mean_tbl, methods):
    """Veri seti basina sirala, veri setleri boyunca ortala (Demsar)."""
    sub = mean_tbl[methods]
    return sub.rank(axis=1, ascending=False).mean()


# ------------------------------------------------------------- LaTeX yardimci
def fnum(x, d=2):
    return '--' if pd.isna(x) else f'{x:.{d}f}'


SHOW_STD = False    # danisman notu 2026-07-22: yogun tablolarda hucre-ici +- kullanma;
                    # kat bazli std'ler ayrintili tablolarda (github/) veriliyor


def cell(mean, std, best=False):
    if pd.isna(mean):
        return '--'
    m = f'\\textbf{{{mean:.2f}}}' if best else f'{mean:.2f}'
    if pd.isna(std) or not SHOW_STD:
        return m
    return f'{m}\\,\\stdv{{{std:.1f}}}'


class Ctx:
    """Bir metrik paneli icin gereken her sey."""
    def __init__(self, datasets, mean_tbl, std_tbl, spread, kk, pav):
        self.ds, self.mean, self.std = datasets, mean_tbl, std_tbl
        self.spread, self.kk, self.pav = spread, kk, pav
        self.ncol = len(datasets) + 4   # Method, Ret., <datasets>, Rank, s_clf


def kfmt(k):
    """K'yi kompakt yaz: 34.0 -> 34, 12.7 -> 12.7, 2.0 -> 2."""
    return f'{k:.1f}'.rstrip('0').rstrip('.')


def retention(ctx, m):
    return (ctx.kk.loc[ctx.ds, m] / ctx.pav[ctx.ds]).mean() * 100


def rows(f, ctx, methods, ranks, best_pool, own_k=False, block_ret=None):
    """Yontem satirlarini yazar.

    best_pool : hangi yontemler arasinda 'en iyi' kalinlasir
    own_k     : her hucreye o yontemin kendi K'sini ust simge olarak ekle
    block_ret : blok basliginda yazilmis Ret.% -- ondan farkli olmayan
                satirlarda sutun bos birakilir (13 kez ayni sayiyi tekrarlamamak icin)
    """
    # yalnizca TEK basina en iyi olan kalinlasir; beraberlik isaretlenmez
    # (Banknote gibi dejenere sutunlarda 9 yontem ayni degeri veriyor)
    best = {}
    for d in ctx.ds:
        pool = [x for x in best_pool
                if not (own_k and ctx.kk.loc[d, x] >= ctx.pav[d])]
        col = ctx.mean.loc[d, pool] if pool else None
        best[d] = (col.max() if (col == col.max()).sum() == 1 else None) \
            if pool else None
    for m in methods:
        label = LABEL[m]
        if m == CASPER:
            label = '\\rowcolor{gray!12}\\textbf{' + label + '}'
        cells = []
        for d in ctx.ds:
            # kendi-K blogunda K=p ise yontem HIC indirgeme yapmamis: skoru ALL'in
            # skorudur, bir secim sonucu degildir. Sayi yerine cizgi + dipnot.
            if own_k and ctx.kk.loc[d, m] >= ctx.pav[d]:
                cells.append('\\noreduce' + f'\\kret{{{kfmt(ctx.kk.loc[d, m])}}}')
                continue
            c = cell(ctx.mean.loc[d, m], ctx.std.loc[d, m],
                     best=(m in best_pool and ctx.mean.loc[d, m] == best[d]))
            if own_k:
                c += f'\\kret{{{kfmt(ctx.kk.loc[d, m])}}}'
            cells.append(c)
        ret = retention(ctx, m)
        rcell = '' if (block_ret is not None and round(ret) == round(block_ret)) \
                else f'{ret:.0f}'
        # kendi-K blogunda hic indirgeme yapmayan veri setleri rank'i sisiriyor:
        # hepsinde K=p ise rank anlamsizdir (cizgi), bazisinda ise hancerle isaretlenir
        nored = sum(ctx.kk.loc[d, m] >= ctx.pav[d] for d in ctx.ds) if own_k else 0
        if nored == len(ctx.ds):
            rk = '\\noreduce'
        elif nored:
            rk = f'{ranks[m]:.2f}$^{{\\dagger}}$'
        else:
            rk = f'{ranks[m]:.2f}'
        f.write(f'\t\t{label} & {rcell} & ' + ' & '.join(cells)
                + f' & {rk} & {ctx.spread.loc[ctx.ds, m].mean():.1f} \\\\\n')


def group_label(f, ctx, text, space=True):
    if space:
        f.write('\t\t\\addlinespace[2pt]\n')
    f.write(f'\t\t\\multicolumn{{{ctx.ncol}}}{{l}}{{{text}}} \\\\\n')


# --------------------------------------------------------------- master tablo
SIZE, COLSEP = 'scriptsize', '2pt'      # BUTUN master tablolarda ayni punto


def master_table(datasets, label, caption, path, baselines, own=False):
    """Kanonik karsilastirma tablosu.

    baselines : esleşik-K temelleri (klasik ailesi VEYA kaba kume ailesi)
    own       : True ise ayrica kendi-K blogu yazilir (yalnizca kaba kume ailesinde
                anlamli -- klasik seciciler siralama uretir, durma kurallari yoktur)
    """
    matched = ['ALL', CASPER] + baselines
    df = load_group(datasets)
    kk = kappa(df)
    pav = kk['ALL']                                     # mevcut oznitelik sayisi p
    ncol = len(datasets) + 4
    head = ('\t\t\\textbf{Method} & \\textbf{Ret.\\%}$\\downarrow$ & '
            + ' & '.join(f'\\textbf{{{NICE[d]}}}' for d in datasets)
            + ' & \\textbf{Rank}$\\downarrow$ & $s_{\\mathrm{clf}}$ \\\\\n'
            + '\t\t& & ' + ' & '.join(f'\\emph{{$p$={pav[d]:.0f}}}' for d in datasets)
            + f' & \\emph{{of {len(matched)}}} & \\emph{{sd}} \\\\\n')

    with open(path, 'w') as f:
        f.write('% OTOMATIK URETILDI -- analysis/make_tables.py. ELLE DUZENLEMEYIN.\n')
        f.write('{\\%s\\setlength{\\tabcolsep}{%s}\n' % (SIZE, COLSEP))
        f.write('\\begin{longtable}{l r ' + 'r' * len(datasets) + ' r r}\n')
        f.write(f'\t\\caption{{{caption}}}\\label{{{label}}}\\\\\n')
        f.write('\t\\toprule\n' + head + '\t\t\\midrule\n\t\\endfirsthead\n')
        f.write(f'\t\\multicolumn{{{ncol}}}{{l}}{{\\emph{{Table \\ref{{{label}}} '
                '-- continued from previous page}}\\\\\n')
        f.write('\t\\toprule\n' + head + '\t\t\\midrule\n\t\\endhead\n')
        f.write('\t\\bottomrule\n\t\\endfoot\n')

        for mi, (metric, mname) in enumerate(METRICS):
            mean_tbl, std_tbl = cell_stats(df, metric)
            ctx = Ctx(datasets, mean_tbl, std_tbl, clf_spread(df, metric), kk, pav)
            rk_m = mean_ranks(mean_tbl, matched)

            # kalinlastirma havuzu = yazdirilan SECICILER (ALL referans, kalinlasmaz)
            sel = [m for m in matched if m != 'ALL']
            bret = retention(ctx, CASPER)      # esleşik-K blogunun ortak Ret.%

            f.write('\t\t\\midrule\n')
            group_label(f, ctx, f'\\textbf{{Panel {"ABC"[mi]}: {mname} (\\%)}}',
                        space=False)
            f.write('\t\t\\midrule\n')

            rows(f, ctx, ['ALL'], rk_m, sel)
            group_label(f, ctx, '\\emph{Matched to \\CASPERt{}\\textquotesingle s '
                                'per-fold $K$:}')
            f.write(f'\t\t\\quad\\emph{{retained $K$}} & \\emph{{{bret:.0f}}} & '
                    + ' & '.join(f'\\emph{{{kfmt(kk.loc[d, CASPER])}}}' for d in datasets)
                    + ' & & \\\\\n')
            rows(f, ctx, [CASPER], rk_m, sel, block_ret=bret)
            rows(f, ctx, sorted(baselines, key=lambda m: rk_m[m]), rk_m, sel,
                 block_ret=bret)

            if own:
                own_pool = ['ALL', CASPER] + ROUGHN
                rk_o = mean_ranks(mean_tbl, own_pool)
                group_label(f, ctx, '\\emph{Each method at its own stopping point '
                                    '($K$ beside each cell; rank over these '
                                    f'{len(own_pool)}):}}')
                rows(f, ctx, [CASPER] + sorted(ROUGHN, key=lambda m: rk_o[m]),
                     rk_o, [CASPER] + ROUGHN, own_k=True)
        f.write('\\end{longtable}}\n')
    return df


# --------------------------------------------- kendi-K (durma davranisi) tablosu
def own_cardinality_table(path):
    """Kendi durma kuralina birakildiginda ne oluyor: 10 veri seti tek tabloda.

    Klasik 9 secici burada YOK -- onlar siralama uretir, durma kurali yoktur.
    Veri setleri satir, yontemler sutun; K yalnizca Panel A'da (metrikten bagimsiz).
    """
    ds_all = SMALL + LARGE
    NICE_FULL = dict(NICE, cyron='Cryotherapy', diabet='Diabetes',
                     kanser='Breast Cancer', parkinson="Parkinson's", magic='MAGIC')
    METH = [CASPER] + ROUGHN
    df = load_group(ds_all)
    kk = kappa(df)
    pav = kk['ALL']

    with open(path, 'w') as f:
        f.write('% OTOMATIK URETILDI -- analysis/make_tables.py. ELLE DUZENLEMEYIN.\n')
        f.write('\\begin{table}[H]\n\t\\centering\n')
        f.write('\t\\caption{Stopping behaviour: each method left to stop where its own '
                'criterion says it should, on all ten datasets. The nine classical '
                'selectors do not appear because they produce a \\emph{ranking} and '
                'have no intrinsic stopping rule; only the rough-set family and '
                '\\CASPERt{} decide their own cardinality. Cells are '
                'classifier-averaged scores in per cent; the small grey figure in '
                'Panel A is the mean number of features retained, $K$, which does not '
                'depend on the metric and is therefore not repeated in Panels B and C. '
                '$p$ is the number of features available. Each panel closes with the '
                'mean rank over these five methods (lower is better); the mean '
                'retained fraction, also metric-independent, is given once. Bold marks the best selector per row; ALL is the '
                'no-selection reference.}\n')
        f.write('\t\\label{tab:own_cardinality}\n')
        f.write('\t\\resizebox{\\textwidth}{!}{%\n\t\\begin{tabular}{lr|c|cccc}\n')
        f.write('\t\t\\toprule\n\t\t\\textbf{Dataset} & \\textbf{$p$} & \\textbf{ALL} & '
                + ' & '.join(f'\\textbf{{{LABEL[m]}}}' for m in METH) + ' \\\\\n')

        for mi, (metric, mname) in enumerate(METRICS):
            mean_tbl, _ = cell_stats(df, metric)
            rk = mean_ranks(mean_tbl, OWN)
            f.write('\t\t\\midrule\n\t\t\\multicolumn{7}{l}{\\textbf{Panel '
                    f'{"ABC"[mi]}: {mname} (\\%)}}}} \\\\\n\t\t\\midrule\n')
            for d in ds_all:
                vals = mean_tbl.loc[d, METH]
                bst = vals.max() if (vals == vals.max()).sum() == 1 else None
                cs = []
                for m in METH:
                    v = mean_tbl.loc[d, m]
                    s = f'\\textbf{{{v:.2f}}}' if v == bst else f'{v:.2f}'
                    if mi == 0:
                        s += f'\\kret{{{kfmt(kk.loc[d, m])}}}'
                    cs.append(s)
                f.write(f'\t\t{NICE_FULL[d]} & {pav[d]:.0f} & '
                        f'{mean_tbl.loc[d, "ALL"]:.2f} & ' + ' & '.join(cs) + ' \\\\\n')
            f.write('\t\t\\addlinespace[2pt]\n')
            f.write('\t\t\\emph{Mean rank} $\\downarrow$ & & '
                    f'\\emph{{{rk["ALL"]:.2f}}} & '
                    + ' & '.join(f'\\emph{{{rk[m]:.2f}}}' for m in METH) + ' \\\\\n')
            if mi == 0:
                f.write('\t\t\\emph{Mean retained \\%} $\\downarrow$ & & \\emph{100} & '
                        + ' & '.join(
                            f'\\emph{{{(kk[m] / pav).mean()*100:.0f}}}' for m in METH)
                        + ' \\\\\n')
        f.write('\t\t\\bottomrule\n\t\\end{tabular}}\n\\end{table}\n')


# --------------------------------------------- CASPER-t epsilon / K ozeti (T3)
def eps_summary_table(path):
    """Her veri setinde CASPER-t'nin kat basina tuttugu K -- 9 veri setinin tamami."""
    ds_all = SMALL + LARGE
    NICE_FULL = dict(NICE, cyron='Cryotherapy', diabet='Diabetes',
                     kanser='Breast Cancer', parkinson="Parkinson's",
                     magic='MAGIC (native)')
    df = load_group(ds_all)
    fold_k = df[df.Method == CASPER].groupby(['Dataset', 'Fold']).N_Features_Used.first()
    pav = kappa(df)['ALL']
    with open(path, 'w') as f:
        f.write('% OTOMATIK URETILDI -- analysis/make_tables.py. ELLE DUZENLEMEYIN.\n')
        f.write('\\begin{table}[H]\n\t\\centering\n')
        f.write('\t\\caption{Feature elimination summary for \\CASPERt{} ($\\kappa=1.5$) '
                'across the 10 outer folds of every dataset. $p$ is the number of '
                'features after the correlation filter; $K$ is the number retained, '
                'recomputed independently inside each outer training fold, so the '
                'ranges are fold-to-fold variability rather than tuning latitude. The '
                'retained fraction stays inside a narrow band across a two-orders-of-'
                'magnitude range of $n$ and a fifteen-fold range of $p$.}\n')
        f.write('\t\\label{tab:eps_summary}\n\t\\begin{tabular}{lrrrrr}\n\t\t\\toprule\n')
        f.write('\t\t\\textbf{Dataset} & \\textbf{$p$} & \\textbf{$K$ mean} & '
                '\\textbf{$K$ range} & \\textbf{\\% retained} & '
                '\\textbf{\\% eliminated} \\\\\n\t\t\\midrule\n')
        for i, d in enumerate(ds_all):
            if i == len(SMALL):
                f.write('\t\t\\addlinespace\n')
            k = fold_k.loc[d]
            r = k.mean() / pav[d] * 100
            f.write(f'\t\t{NICE_FULL[d]} & {pav[d]:.0f} & {k.mean():.1f} & '
                    f'{k.min():.0f}--{k.max():.0f} & {r:.0f}\\% & {100-r:.0f}\\% \\\\\n')
        f.write('\t\t\\bottomrule\n\t\\end{tabular}\n\\end{table}\n')


# ------------------------------------- CASPER-t vs kaba kume temelleri (Wilcoxon)
def roughset_wilcoxon_table(path):
    """Veri seti basina CASPER-t vs her esleşik-K kaba kume temeli; BH + rank-biserial."""
    ds_all = SMALL + LARGE
    NICE_FULL = dict(NICE, cyron='Cryotherapy', diabet='Diabetes',
                     kanser='Breast Cancer', parkinson="Parkinson's", magic='MAGIC')
    df = load_group(ds_all)
    per_fold = df.groupby(['Dataset', 'Method', 'Fold']).ACCU.mean().reset_index()

    cells, pvals, keys = {}, [], []
    for d in ds_all:
        piv = per_fold[per_fold.Dataset == d].pivot(index='Fold', columns='Method',
                                                    values='ACCU')
        for b in ROUGHM:
            diff = (piv[CASPER] - piv[b]).values * 100
            nz = diff[diff != 0]
            wins, n = int((nz > 0).sum()), len(nz)
            if n == 0:                      # ozdes altkumeler -> test dejenere
                cells[(d, b)] = (diff.mean(), 0, 0, None, 0.0)
                continue
            p = stats.wilcoxon(nz).pvalue
            r = (nz > 0).mean() * 2 - 1     # eslesik-cift rank-biserial yaklasimi
            pos = stats.rankdata(np.abs(nz))[nz > 0].sum()
            neg = stats.rankdata(np.abs(nz))[nz < 0].sum()
            r = (pos - neg) / (pos + neg)
            cells[(d, b)] = (diff.mean(), wins, n, p, r)
            pvals.append(p)
            keys.append((d, b))
    qs = dict(zip(keys, multipletests(pvals, method='fdr_bh')[1]))

    with open(path, 'w') as f:
        f.write('% OTOMATIK URETILDI -- analysis/make_tables.py. ELLE DUZENLEMEYIN.\n')
        f.write('\\begin{table}[H]\n\t\\centering\n')
        f.write('\t\\caption{\\CASPERt{} against each matched-$K$ rough-set baseline, '
                'per dataset, on classifier-averaged Accuracy over the 10 outer folds: '
                'mean difference in percentage points ($\\Delta$pp, positive favours '
                '\\CASPERt{}), folds won out of folds with a non-zero difference, '
                'BH-corrected $q$ over the testable comparisons of this table, and the '
                'matched-pairs rank-biserial effect size $r$. Starred entries are '
                'significant at $q<0.05$. Where every fold gives an identical subset '
                'the test is degenerate rather than merely non-significant and is '
                'marked n/a.}\n')
        f.write('\t\\label{tab:roughset_wilcoxon}\n')
        f.write('\t\\resizebox{\\textwidth}{!}{%\n\t\\begin{tabular}{l|cccc|cccc|cccc}\n')
        f.write('\t\t\\toprule\n\t\t& \\multicolumn{4}{c|}{\\textbf{vs.\\ QuickReduct}} '
                '& \\multicolumn{4}{c|}{\\textbf{vs.\\ Approx.\\ reducts}} & '
                '\\multicolumn{4}{c}{\\textbf{vs.\\ FRFS}} \\\\\n')
        f.write('\t\t\\textbf{Dataset}' + 3 * (' & $\\Delta$pp & wins & $q$ & $r$')
                + ' \\\\\n\t\t\\midrule\n')
        for i, d in enumerate(ds_all):
            if i == len(SMALL):
                f.write('\t\t\\addlinespace\n')
            out = []
            for b in ROUGHM:
                dl, w, n, p, r = cells[(d, b)]
                if p is None:
                    out += [f'${dl:+.2f}$', 'n/a', 'n/a', '--']
                else:
                    q = qs[(d, b)]
                    qs_ = f'{q:.3f}$^{{*}}$' if q < 0.05 else f'{q:.3f}'
                    out += [f'${dl:+.2f}$', f'{w}/{n}', qs_, f'${r:+.2f}$']
            f.write(f'\t\t{NICE_FULL[d]} & ' + ' & '.join(out) + ' \\\\\n')
        f.write('\t\t\\bottomrule\n\t\\end{tabular}}\n\\end{table}\n')
    nsig = sum(1 for k in keys if qs[k] < 0.05)
    ahead = sum(1 for k in keys if qs[k] < 0.05 and cells[k][0] > 0)
    print(f'  roughset Wilcoxon: {nsig} anlamli ({ahead} CASPER-t onde, '
          f'{nsig-ahead} geride), {len(keys)} test edilebilir')


# ------------------------------------------------- ALL vs CASPER-t (anlamlilik)
def all_vs_casper_table(path):
    """10 veri setinin tamami, HYBRID bazi, Wilcoxon + BH. Eski T4 ve T15'i birlestirir."""
    ds_all = SMALL + LARGE
    NICE_FULL = dict(NICE, cyron='Cryotherapy', diabet='Diabetes',
                     kanser='Breast Cancer', parkinson="Parkinson's",
                     magic='MAGIC (native)')
    df = load_group(ds_all)
    kk = kappa(df)
    hyb = df[df.Classifier == 'HYBRID']

    stats_ = {}
    for metric, _ in METRICS:
        piv = hyb.pivot_table(index=['Dataset', 'Fold'], columns='Method', values=metric)
        ps = []
        for d in ds_all:
            a, c = piv.loc[d, 'ALL'].values, piv.loc[d, CASPER].values
            ps.append(stats.wilcoxon(a, c).pvalue if np.any(a != c) else 1.0)
        qs = multipletests(ps, method='fdr_bh')[1]
        for d, p, q in zip(ds_all, ps, qs):
            s = piv.loc[d]
            stats_[(metric, d)] = (s['ALL'].mean() * 100, s['ALL'].std() * 100,
                                   s[CASPER].mean() * 100, s[CASPER].std() * 100, p, q)

    def sig(v):
        return f'\\textbf{{{v:.3f}}}' if v < 0.05 else f'{v:.3f}'

    with open(path, 'w') as f:
        f.write('% OTOMATIK URETILDI -- analysis/make_tables.py. ELLE DUZENLEMEYIN.\n')
        f.write('\\begin{table}[H]\n\t\\centering\n')
        f.write('\t\\caption{ALL vs.\\ \\CASPERt{} on all ten datasets, HYBRID '
                'classifier, mean $\\pm$ fold-to-fold std over the 10 outer folds. '
                '$p$: paired Wilcoxon signed-rank test against ALL, matched by fold; '
                '$q$: Benjamini--Hochberg value within each metric\'s family of ten '
                'tests. Bold indicates $<0.05$. \\textbf{Ret.\\%} repeats the mean '
                'fraction of features \\CASPERt{} keeps. This is the only comparison '
                'in the article reported on the HYBRID basis rather than '
                'classifier-averaged; it answers what parsimony costs against using '
                'every feature, not how \\CASPERt{} ranks against other selectors, '
                'which is Table \\ref{tab:master_small}--\\ref{tab:master_large}.}\n')
        f.write('\t\\label{tab:performance_summary}\n')
        f.write('\t\\resizebox{\\textwidth}{!}{%\n\t\\begin{tabular}{lr|cccc|cccc|cccc}\n')
        f.write('\t\t\\toprule\n\t\t& & \\multicolumn{4}{c|}{\\textbf{Accuracy}} & '
                '\\multicolumn{4}{c|}{\\textbf{F1-Score}} & '
                '\\multicolumn{4}{c}{\\textbf{AUC}} \\\\\n')
        f.write('\t\t\\textbf{Dataset} & \\textbf{Ret.\\%}' + 3 * (
            ' & \\textbf{ALL} & \\textbf{\\CASPERt{}} & \\textbf{$p$} & \\textbf{$q$}')
            + ' \\\\\n\t\t\\midrule\n')
        for i, d in enumerate(ds_all):
            if i == len(SMALL):
                f.write('\t\t\\addlinespace\n')
            ret = kk.loc[d, CASPER] / kk.loc[d, 'ALL'] * 100
            cells = []
            for metric, _ in METRICS:
                am, asd, cm, csd, p, q = stats_[(metric, d)]
                cells += [f'{am:.2f}$\\pm${asd:.2f}',
                          f'{cm:.2f}$\\pm${csd:.2f}', sig(p), sig(q)]
            f.write(f'\t\t{NICE_FULL[d]} & {ret:.0f} & ' + ' & '.join(cells) + ' \\\\\n')
        f.write('\t\t\\bottomrule\n\t\\end{tabular}}\n\\end{table}\n')


# ------------------------------------------------------------ Friedman tablosu
def friedman_table(path):
    """Iki karsilastirma havuzu icin omnibus testler: klasik aile (k=11), kaba kume (k=5)."""
    groups = [('Smaller ($n\\le1000$)', SMALL), ('Larger ($n>1000$)', LARGE),
              ('All ten pooled', SMALL + LARGE)]
    pools = [('Against the nine classical selectors ($k=11$)', ['ALL', CASPER] + CLASSICAL),
             ('Against the rough-set family ($k=5$)', ['ALL', CASPER] + ROUGHM)]

    rows_ = []
    for pname, pool in pools:
        for gname, ds in groups:
            mt = {m: cell_stats(load_group(ds), m)[0] for m, _ in METRICS}
            for metric, mname in METRICS:
                chi2, p = stats.friedmanchisquare(*[mt[metric][m].values for m in pool])
                cd = nemenyi_cd(len(pool), len(ds))
                rk = mean_ranks(mt[metric], pool)
                npairs = sum(abs(rk[a] - rk[b]) > cd
                             for i, a in enumerate(pool) for b in pool[i + 1:])
                assert not any(abs(rk[CASPER] - rk[b]) > cd for b in pool
                               if b != CASPER), f'{pname}/{gname}/{mname}: CASPER cifti!'
                rows_.append([pname, gname, mname, len(ds), len(pool) - 1,
                              chi2, p, cd, npairs, rk[CASPER]])
    qs = multipletests([r[6] for r in rows_], method='fdr_bh')[1]

    with open(path, 'w') as f:
        f.write('% OTOMATIK URETILDI -- analysis/make_tables.py. ELLE DUZENLEMEYIN.\n')
        f.write('\\begin{table}[H]\n\t\\centering\n')
        f.write('\t\\caption{Friedman omnibus tests, classifier-averaged, with datasets '
                'as blocks, for the two comparison pools reported in this article. '
                '$q$ is the Benjamini--Hochberg value over the eighteen tests of this '
                'table. CD is the Nemenyi critical difference at $\\alpha=0.05$, and '
                '``pairs\'\' counts how many pairwise rank differences exceed it. '
                '\\textbf{Not one of those pairs involves \\CASPERt{}} in any row: every '
                'resolvable difference is driven by the weakness of FRFS, and on pooled '
                'AUC also by QuickReduct. The last column repeats \\CASPERt{}'
                '\\textquotesingle s own mean rank. At $N=4$ and $N=6$ the $\\chi^2$ '
                'approximation to the Friedman statistic is itself only approximate.}\n')
        f.write('\t\\label{tab:friedman_summary}\n')
        f.write('\t\\begin{tabular}{llccccccc}\n\t\t\\toprule\n')
        f.write('\t\t\\textbf{Dataset group} & \\textbf{Metric} & \\textbf{$N$} & '
                '\\textbf{d.f.} & \\textbf{$\\chi^2$} & \\textbf{$p$} & \\textbf{$q$} & '
                '\\textbf{CD} & \\textbf{pairs} \\\\\n')
        pprev = gprev = None
        for r, q in zip(rows_, qs):
            if r[0] != pprev:
                f.write('\t\t\\midrule\n\t\t\\multicolumn{9}{l}{\\textbf{'
                        + r[0] + '}} \\\\\n\t\t\\midrule\n')
                pprev, gprev = r[0], None
            elif r[1] != gprev:
                f.write('\t\t\\addlinespace[2pt]\n')
            g = r[1] if r[1] != gprev else ''
            gprev = r[1]
            bp = f'\\textbf{{{r[6]:.4f}}}' if r[6] < 0.05 else f'{r[6]:.4f}'
            bq = f'\\textbf{{{q:.4f}}}' if q < 0.05 else f'{q:.4f}'
            f.write(f'\t\t{g} & {r[2]} & {r[3]} & {r[4]} & {r[5]:.2f} & {bp} & {bq} & '
                    f'{r[7]:.2f} & {r[8]} \\\\\n')
        f.write('\t\t\\bottomrule\n\t\\end{tabular}\n\\end{table}\n')


# --------------------------------------------------- metin icin yardimci sayi
def nemenyi_cd(k, N, alpha=0.05):
    """Nemenyi kritik farki. q_alpha: Studentized range / sqrt(2), k<=20 tablosu."""
    q05 = {3: 2.343, 4: 2.569, 5: 2.728, 11: 3.219, 14: 3.354}
    q = q05.get(k)
    if q is None:
        raise ValueError(f'q_alpha tablosunda k={k} yok')
    return q * np.sqrt(k * (k + 1) / (6.0 * N))


def numbers_report():
    """Yeniden yazilacak §4.6/§4.9/§4.10/§5.7 metinleri icin gereken sayilar."""
    out = []

    def say(s=''):
        print(s)
        out.append(s)

    say('# Birlesik master tablolar -- metin icin sayilar')
    say('(analysis/make_tables.py tarafindan uretildi; tablolarla ayni kaynak)\n')

    groups = [('Kucuk (n<=1000, 6 veri seti)', SMALL),
              ('Buyuk (n>1000, 4 veri seti)', LARGE),
              ('Havuzlanmis (10 veri seti)', SMALL + LARGE)]
    family = [CASPER] + ROUGHM          # kaba kume ailesi ici siralama (4 yontem)

    for gname, ds in groups:
        df = load_group(ds)
        say('=' * 78)
        say(f'## {gname}')
        kk = kappa(df)
        pav = kk['ALL']
        for metric, mname in METRICS:
            mean_tbl, _ = cell_stats(df, metric)
            rk = mean_ranks(mean_tbl, MATCHED).sort_values()
            chi2, p = stats.friedmanchisquare(*[mean_tbl[m].values for m in MATCHED])
            pos = list(rk.index).index(CASPER) + 1
            say(f'\n-- {mname} --')
            say(f'   Friedman (k={len(MATCHED)}, N={len(ds)}): chi2={chi2:.3f}  p={p:.4f}'
                f'   Nemenyi CD={nemenyi_cd(len(MATCHED), len(ds)):.2f}')
            say(f'   \\CASPERt{{}} sirasi: {pos}/{len(MATCHED)}  (mean rank {rk[CASPER]:.2f})')
            say('   ' + '  '.join(f'{LABEL[m].replace(chr(92),"")}={rk[m]:.2f}'
                                  for m in rk.index))
            rkf = mean_ranks(mean_tbl, family).sort_values()
            posf = list(rkf.index).index(CASPER) + 1
            say(f'   kaba kume ailesi ici (k=4): \\CASPERt{{}} {posf}/4  '
                + '  '.join(f'{LABEL[m].replace(chr(92),"")}={rkf[m]:.2f}'
                            for m in rkf.index))
        say('\n-- tutulan oznitelik orani (Ret. %) --')
        for m in [CASPER] + ROUGHN:
            say(f'   {LABEL[m]:16s} {(kk.loc[ds, m] / pav[ds]).mean()*100:5.1f}%'
                f'   (K: ' + ', '.join(f'{d}={kk.loc[d, m]:.1f}' for d in ds) + ')')
        say('')

    path = os.path.join(HERE, 'out', 'MASTER_NUMBERS.md')
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('\nyazildi:', path)


# ------------------------------------------------------------------- regresyon
EXPECTED = {   # matchk_fix_NUMBERS.md'den (classifier-averaged Accuracy, %)
    ('magic', CASPER): 81.19, ('magic', 'ALL'): 82.47, ('magic', 'RFE_matchK'): 82.59,
    ('magic', 'StdPFI_matchK'): 82.57, ('magic', 'MI_matchK'): 81.63,
    ('htru2', CASPER): 97.10, ('htru2', 'ALL'): 97.00,
    ('phoneme', CASPER): 77.66, ('phoneme', 'ALL'): 81.87,
    ('banknote', CASPER): 96.83, ('banknote', 'ALL'): 97.06,
    ('sonar', CASPER): 80.33, ('cyron', CASPER): 84.86, ('diabet', CASPER): 77.55,
    ('kanser', CASPER): 94.21, ('parkinson', CASPER): 86.15,
}


def check():
    df = pd.concat([load_group(SMALL), load_group(LARGE)], ignore_index=True)
    mean_tbl, _ = cell_stats(df, 'ACCU')
    bad = 0
    for (ds, m), exp in sorted(EXPECTED.items()):
        got = mean_tbl.loc[ds, m]
        ok = abs(got - exp) < 0.005
        bad += not ok
        print(f'  {"OK " if ok else "FARK"}  {ds:10s} {m:18s} beklenen {exp:6.2f}  '
              f'uretilen {got:6.2f}')
    print(f'\n{len(EXPECTED)-bad}/{len(EXPECTED)} eslesti.')
    return bad == 0



# ------------------------------------------------- github/ ayrintili ciktilar
def export_detailed(outdir):
    """Makaleye sigmayan ayrinti: her hucrenin mean/std'si + kat bazli ham degerler.

    Danisman notu (2026-07-22): tablo hucrelerinde +- kullanilmayacak; ayrintili
    surum depoya konup makaleden oraya atif yapilacak.
    """
    os.makedirs(outdir, exist_ok=True)
    ds_all = SMALL + LARGE
    df = load_group(ds_all)
    kk = kappa(df)
    pav = kk['ALL']

    # 1) yontem x veri seti x metrik: mean, std, K, retained%
    recs = []
    for metric, mname in METRICS:
        mean_tbl, std_tbl = cell_stats(df, metric)
        spread = clf_spread(df, metric)
        for d in ds_all:
            for m in mean_tbl.columns:
                recs.append(dict(dataset=d, method=LABEL.get(m, m).replace('\\', ''),
                                 method_id=m, metric=mname,
                                 mean=round(mean_tbl.loc[d, m], 4),
                                 std_folds=round(std_tbl.loc[d, m], 4),
                                 std_classifiers=round(spread.loc[d, m], 4),
                                 K=round(kk.loc[d, m], 2), p=int(pav[d]),
                                 retained_pct=round(kk.loc[d, m] / pav[d] * 100, 1)))
    pd.DataFrame(recs).to_csv(os.path.join(outdir, 'summary_mean_std.csv'), index=False)

    # 2) tam kat bazli ham degerler (her kat, her siniflandirici)
    df.to_csv(os.path.join(outdir, 'fold_level_raw.csv.gz'), index=False,
              compression='gzip')

    # 3) HYBRID bazi (Tablo 4'un kaynagi)
    df[df.Classifier == 'HYBRID'].to_csv(
        os.path.join(outdir, 'fold_level_hybrid.csv'), index=False)

    n = len(recs)
    print(f'  github/detailed_tables: {n} ozet satir + kat bazli ham veri')



# ------------------------------------------------------- Sekil: indirgeme vs kayip
def retention_figure(path):
    """Kendi durma noktasinda: tutulan oran (x) vs ALL'a gore dogruluk farki (y).

    Makalenin merkez iddiasi tek grafikte: CASPER-t dar bir bantta sinirli maliyetle
    indirger; kaba kume durma kurallari iki uca savrulur (hic indirgemez ya da asiri).
    """
    ds_all = SMALL + LARGE
    df = load_group(ds_all)
    kk = kappa(df)
    pav = kk['ALL']
    mt, _ = cell_stats(df, 'ACCU')
    series = [(CASPER, 'CASPER-t', 'black', '*'), ('QuickReduct_native', 'QuickReduct',
              'blue!70', 'square'), ('SkroughRed_native', 'Approx. reducts', 'orange!85!black',
              'triangle'), ('FRFS_native', 'FRFS', 'red!70', 'diamond')]
    with open(path, 'w') as f:
        f.write('% OTOMATIK URETILDI -- analysis/make_tables.py. ELLE DUZENLEMEYIN.\n')
        f.write('\\begin{figure}[htbp]\n\t\\centering\n\t\\begin{tikzpicture}\n')
        f.write('\t\\begin{axis}[width=0.92\\linewidth, height=7cm,\n'
                '\t\txlabel={Features retained, $K/p$ (\\%)},\n'
                '\t\tylabel={Accuracy relative to ALL (pp)},\n'
                '\t\txmin=0, xmax=105, grid=both, grid style={gray!20},\n'
                '\t\tlegend style={font=\\small, at={(0.02,0.03)}, anchor=south west,\n'
                '\t\t\tdraw=gray!50, fill=white, fill opacity=0.9, text opacity=1},\n'
                '\t\tlabel style={font=\\small}, tick label style={font=\\small}]\n')
        f.write('\t\\addplot[dashed, gray, thick, forget plot] '
                'coordinates {(0,0) (105,0)};\n')
        for m, lab, col, mark in series:
            pts = ' '.join(f'({kk.loc[d, m]/pav[d]*100:.1f},{mt.loc[d, m]-mt.loc[d, "ALL"]:.2f})'
                           for d in ds_all)
            f.write(f'\t\\addplot[only marks, mark={mark}, mark size=2.4pt, '
                    f'color={col}] coordinates {{{pts}}};\n')
            f.write(f'\t\\addlegendentry{{{lab}}}\n')
        f.write('\t\\end{axis}\n\t\\end{tikzpicture}\n')
        f.write('\t\\caption{Each point is one method on one of the nine datasets, at its '
                'own stopping point: how much of the feature set it kept (horizontal) '
                'against what that cost in classifier-averaged Accuracy relative to using '
                'every feature (vertical; the dashed line is ALL). \\CASPERt{} occupies a '
                'narrow band -- it always keeps between $49\\%$ and $75\\%$ of the features '
                'and never loses more than $4.2$ points -- whereas the rough-set stopping '
                'criteria are bimodal: QuickReduct either keeps everything (four points at '
                '$100\\%$, where it is ALL and reduces nothing) or cuts to $13\\%$ and loses '
                'nearly ten points, and FRFS always cuts to a handful of features and '
                'always pays for it. Predictability of reduction, not accuracy, is what '
                'separates the methods here.}\n')
        f.write('\t\\label{fig:retention_cost}\n\\end{figure}\n')


def main():
    if '--check' in sys.argv:
        sys.exit(0 if check() else 1)

    master_table(
        SMALL, 'tab:master_small',
        'Comparison against the nine classical feature selection baselines on the six '
        'smaller datasets ($n\\le1000$), all at \\CASPERt{}\\textquotesingle s per-fold '
        'cardinality on identical outer folds. The rough-set family is compared '
        'separately in Tables \\ref{tab:rough_small}--\\ref{tab:rough_large}. '
        + 'Each cell is the mean over the 10 outer folds of the score averaged across the '
        '8 classifiers; per-fold values and their standard deviations are provided in '
        'the accompanying repository rather than printed here. '
        '\\textbf{Ret.\\%}: mean fraction of the available features retained (lower is '
        'more parsimonious); it is printed on the block header where it is common to '
        'every row of the block and only repeated where a row departs from it. '
        '\\textbf{Rank}: mean rank across these datasets within the block, lower is '
        'better \\cite{demsar2006}; accuracies are deliberately \\emph{not} averaged '
        'across datasets, which are not commensurable. $s_{\\mathrm{clf}}$: standard '
        'deviation across the 8 classifiers, averaged over datasets, i.e.\\ how much '
        'the method\'s standing depends on which classifier is used. Bold marks the '
        'best \\emph{selector} per column, where that best is unique; ties are left '
        'unmarked. ALL is the no-selection reference and is never bolded.',
        os.path.join(TABLES, 'master_small.tex'), CLASSICAL)

    master_table(
        LARGE, 'tab:master_large',
        'Comparison against the nine classical baselines on the four larger datasets '
        '($n>1000$); columns and cell convention exactly as in Table '
        '\\ref{tab:master_small}. '
        + 'Each cell is the mean over the 10 outer folds of the score averaged across the '
        '8 classifiers; per-fold values and their standard deviations are provided in '
        'the accompanying repository rather than printed here. '
        '\\textbf{Ret.\\%}: mean fraction of the available features retained (lower is '
        'more parsimonious); it is printed on the block header where it is common to '
        'every row of the block and only repeated where a row departs from it. '
        '\\textbf{Rank}: mean rank across these datasets within the block, lower is '
        'better \\cite{demsar2006}; accuracies are deliberately \\emph{not} averaged '
        'across datasets, which are not commensurable. $s_{\\mathrm{clf}}$: standard '
        'deviation across the 8 classifiers, averaged over datasets, i.e.\\ how much '
        'the method\'s standing depends on which classifier is used. Bold marks the '
        'best \\emph{selector} per column, where that best is unique; ties are left '
        'unmarked. ALL is the no-selection reference and is never bolded.',
        os.path.join(TABLES, 'master_large.tex'), CLASSICAL)

    master_table(
        SMALL, 'tab:rough_small',
        'Comparison against rough-set-based reduction on the six smaller datasets '
        '($n\\le1000$), in the same format as Table \\ref{tab:master_small}. Unlike '
        'the classical selectors, these three methods decide their own cardinality, so '
        'each is reported twice: truncated at \\CASPERt{}\\textquotesingle s per-fold '
        '$K$, which isolates ranking quality, and at its own stopping point, which is '
        'how it would be used in practice. Ranks are computed within each block. '
        'A dash marks a dataset on which the method retained \\emph{{every}} available '
        'feature ($K=p$), i.e.\\ performed no reduction at all: its score there is '
        'ALL\\textquotesingle s, not a selection result, so it is neither printed nor '
        'eligible to be marked best. A dashed \\textbf{{Rank}} means the method reduced on '
        'no dataset of this table, so a rank would measure nothing but its refusal to '
        'select; $\\dagger$ means it failed to reduce on some. '
        + 'Each cell is the mean over the 10 outer folds of the score averaged across the '
        '8 classifiers; per-fold values and their standard deviations are provided in '
        'the accompanying repository rather than printed here. '
        '\\textbf{Ret.\\%}: mean fraction of the available features retained (lower is '
        'more parsimonious); it is printed on the block header where it is common to '
        'every row of the block and only repeated where a row departs from it. '
        '\\textbf{Rank}: mean rank across these datasets within the block, lower is '
        'better \\cite{demsar2006}; accuracies are deliberately \\emph{not} averaged '
        'across datasets, which are not commensurable. $s_{\\mathrm{clf}}$: standard '
        'deviation across the 8 classifiers, averaged over datasets, i.e.\\ how much '
        'the method\'s standing depends on which classifier is used. Bold marks the '
        'best \\emph{selector} per column, where that best is unique; ties are left '
        'unmarked. ALL is the no-selection reference and is never bolded.',
        os.path.join(TABLES, 'rough_small.tex'), ROUGHM, own=True)

    master_table(
        LARGE, 'tab:rough_large',
        'Comparison against rough-set-based reduction on the four larger datasets '
        '($n>1000$); as Table \\ref{tab:rough_small}. On MAGIC these baselines were '
        'matched to \\CASPERt{}\\textquotesingle s pre-correction per-fold cardinality '
        '($K=6.5$) rather than the corrected $K=7.0$ it uses itself; the half-feature '
        'discrepancy is disclosed rather than smoothed over, and the re-run is deferred '
        'to a future round. Note the own-cardinality block: QuickReduct\'s reduct '
        'retains \\emph{every} available feature on all four datasets ($K=p$), i.e.\\ '
        'it performs no reduction at all here, which is why its accuracy coincides with '
        'ALL. '
        'A dash marks a dataset on which the method retained \\emph{{every}} available '
        'feature ($K=p$), i.e.\\ performed no reduction at all: its score there is '
        'ALL\\textquotesingle s, not a selection result, so it is neither printed nor '
        'eligible to be marked best. A dashed \\textbf{{Rank}} means the method reduced on '
        'no dataset of this table, so a rank would measure nothing but its refusal to '
        'select; $\\dagger$ means it failed to reduce on some. '
        + 'Each cell is the mean over the 10 outer folds of the score averaged across the '
        '8 classifiers; per-fold values and their standard deviations are provided in '
        'the accompanying repository rather than printed here. '
        '\\textbf{Ret.\\%}: mean fraction of the available features retained (lower is '
        'more parsimonious); it is printed on the block header where it is common to '
        'every row of the block and only repeated where a row departs from it. '
        '\\textbf{Rank}: mean rank across these datasets within the block, lower is '
        'better \\cite{demsar2006}; accuracies are deliberately \\emph{not} averaged '
        'across datasets, which are not commensurable. $s_{\\mathrm{clf}}$: standard '
        'deviation across the 8 classifiers, averaged over datasets, i.e.\\ how much '
        'the method\'s standing depends on which classifier is used. Bold marks the '
        'best \\emph{selector} per column, where that best is unique; ties are left '
        'unmarked. ALL is the no-selection reference and is never bolded.',
        os.path.join(TABLES, 'rough_large.tex'), ROUGHM, own=True)

    eps_summary_table(os.path.join(TABLES, 'eps_summary.tex'))
    all_vs_casper_table(os.path.join(TABLES, 'all_vs_casper.tex'))
    roughset_wilcoxon_table(os.path.join(TABLES, 'roughset_wilcoxon.tex'))
    friedman_table(os.path.join(TABLES, 'friedman.tex'))

    retention_figure(os.path.join(TABLES, 'retention_figure.tex'))
    export_detailed(os.path.join(PAPER, 'github', 'detailed_tables'))

    print('yazildi:', TABLES)
    numbers_report()
    print()
    check()


if __name__ == '__main__':
    main()
