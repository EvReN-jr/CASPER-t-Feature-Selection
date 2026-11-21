# -*- coding: utf-8 -*-
"""
Reference cross-validation pipeline of the article (single file).

Runs, on every outer fold of a 5 x 2 repeated stratified cross-validation,
the Jenks discretization fitted on the training fold, the CASPER-t selector
(V1Selector + _tstat_select), the classical baselines, the eight classifiers
and the significance tests, and writes per-dataset and master result files.
The datasets are fetched from OpenML on first use, or read from
DATA_DIR/<name>_READY_FOR_RS.xlsx when present.  Missing Python packages are
installed on import.

Usage:  python gamma_pfi_uhem.py
"""

import os
import sys
import math
import time
import warnings
import subprocess
import importlib

warnings.filterwarnings('ignore')

# =============================================================================
# BOLUM 0 - COLAB BOOTSTRAP: eksik paketleri kur
# =============================================================================

def _ensure_pkg(import_name, pip_name=None):
    pip_name = pip_name or import_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"  [kurulum] {pip_name} bulunamadi, kuruluyor...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)


print("=" * 80)
print("BAGIMLILIKLAR KONTROL EDILIYOR (Colab'da eksik olabilecekler kuruluyor)")
print("=" * 80)
for _imp, _pip in [
    ("numpy", None), ("pandas", None), ("scipy", None), ("sklearn", "scikit-learn"),
    ("openpyxl", None), ("xgboost", None), ("shap", None),
    ("skrebate", None), ("scikit_posthocs", "scikit-posthocs"),
]:
    _ensure_pkg(_imp, _pip)

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon, spearmanr, friedmanchisquare
from sklearn.cluster import KMeans
from sklearn.feature_selection import f_classif, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.datasets import fetch_openml
from skrebate import ReliefF
import shap
import scikit_posthocs as sp
from xgboost import XGBClassifier

# =============================================================================
# SECTION 1 - KONFIGURASYON (orijinal TDA_feature_selection_pipeline.py ile ayni)
# =============================================================================

TARGET_CANDIDATES = [
    'label', 'target', 'Target', 'Class', 'class',
    'Result_of_Treatment', 'status', 'Risk_Class',
]

EXCLUDE_COLS = ['id', 'ID', 'Unnamed: 0', 'index', 'patient_id', 'name']

N_REP        = 30     # Gamma-PFI permutasyon tekrar sayisi
N_EPS_PTS    = 25     # Epsilon tarama nokta sayisi
N_SPLITS_OUTER   = 5  # Nested CV: dis fold sayisi
N_REPEATS_OUTER  = 2  # Nested CV: tekrar sayisi (toplam 10 dis kat)
RANDOM_STATE = 42

MAX_N_TOPOLOGICAL   = 3000
MAX_N_CLASSIFIER_TRAIN = 4000
RELIEF_N_NEIGHBORS  = 50
N_REP_STD_PFI       = 20

JENKS_GVF   = 0.80
JENKS_BINS  = 7

# --- Colab-guvenli yol yonetimi: __file__ bir hucrede tanimsiz olabilir. ---
try:
    _ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _ROOT = '/content/colab_ozel_run'

DATA_DIR    = os.path.join(_ROOT, 'data')
PER_DS_DIR  = os.path.join(_ROOT, 'results', 'per_dataset')
MASTER_DIR  = os.path.join(_ROOT, 'results', 'master')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PER_DS_DIR, exist_ok=True)
os.makedirs(MASTER_DIR, exist_ok=True)

CLASSICAL_SCORE_COLS = [
    'ANOVA_Score', 'MI_Score', 'LASSO_Score', 'RF_Score', 'GB_Score',
    'RFE_Score', 'SHAP_Score', 'ReliefF_Score', 'StdPFI_Score',
]

# --- Veri setleri: magic (oncelikli acik uc) + 4 yeni "1k+" veri seti. ---
# Sira bilincli secildi: banknote once (en kucuk/en hizli -- kurulum/kod
# hatalarini dakikalar icinde yakalamak icin "duman testi"), sonra magic
# (asil hedef -- Colab kopse bile bu bitmis olsun diye erken), sonra htru2
# (ikinci buyuk-n veri seti), en sonda spambase/phoneme (daha kucuk n).
DATASETS = {
    "banknote":  ("banknote_READY_FOR_RS.xlsx", "target"),
    "magic":     ("magicdata3_READY_FOR_RS.xlsx", "target"),
    "htru2":     ("htru2_READY_FOR_RS.xlsx", "target"),
    "spambase":  ("spambase_READY_FOR_RS.xlsx", "target"),
    "phoneme":   ("phoneme_READY_FOR_RS.xlsx", "target"),
}

# Hizli dogrulama icin: RUN_ONLY = ["banknote"] yapip once tek basina deneyin.
# Tam kosum icin None birakin.
# UHeM'de paralel/ayri SLURM islerinden calistirmayi kolaylastirmak icin:
# `python gamma_pfi_uhem.py magic` ya da GAMMA_PFI_DATASETS="magic,htru2"
# ortam degiskeniyle de gecilebilir -- ikisi de yoksa None (tum veri setleri).
RUN_ONLY = None
if len(sys.argv) > 1:
    RUN_ONLY = [a.strip() for a in sys.argv[1:] if a.strip()]
elif os.environ.get("GAMMA_PFI_DATASETS"):
    RUN_ONLY = [a.strip() for a in os.environ["GAMMA_PFI_DATASETS"].split(",") if a.strip()]


# =============================================================================
# SECTION 1b - YENI VERI SETLERI: OpenML'den hazirlama (idempotent)
# =============================================================================

# OpenML'deki TAM/dogru veri seti adlari -- OpenML'in resmi arama API'sinde
# (number_instances fingerprint ile) tek tek dogrulandi (bkz. gelistirme
# notlari): "HTRU2" adiyla kayitli DEGIL, "Pulsar-Dataset-HTRU2" (did=45558).
# "MagicTelescope" (did=1120) dogrulandi: n=19020, p=11 numerik+1 sinif --
# makaledeki magicdata3_READY_FOR_RS.xlsx ile AYNI HAM VERI (UCI MAGIC Gamma
# Telescope), ama makaledeki on-isleme (Winsorization/korelasyon-filtreleme)
# BU dosyada YOK -- bkz. asagidaki not.
OPENML_SPECS = {
    "banknote": "banknote-authentication",
    "phoneme":  "phoneme",
    "spambase": "spambase",
    "htru2":    "Pulsar-Dataset-HTRU2",
    "magic":    "MagicTelescope",
}


def _fetch_and_prepare_openml(ds_key, openml_name, out_path):
    if os.path.exists(out_path):
        print(f"  [{ds_key}] zaten mevcut, yeniden indirilmiyor: {out_path}")
        return
    print(f"  [{ds_key}] OpenML'den cekiliyor (name='{openml_name}')...")
    try:
        bunch = fetch_openml(name=openml_name, version=1, as_frame=True, parser='auto')
    except Exception as e:
        print(f"    version=1 basarisiz ({e!r}); version belirtilmeden yeniden deneniyor...")
        bunch = fetch_openml(name=openml_name, as_frame=True, parser='auto')

    X = bunch.data.copy()
    y = bunch.target

    # Savunma amacli #1: bazi OpenML aynalarinda 'default_target_attribute'
    # metadata'si eksik olabilir (bu calismada HTRU2/Pulsar-Dataset-HTRU2'de
    # gozlemlendi, dogrulandi -- did=45558, version=2 -- OpenML'in kendi
    # kaynak dosyasindaki ARFF baslik satiri eksik). Bu durumda fetch_openml
    # target=None doner VE ilk veri satirini yanlislikla sutun adi olarak
    # kullanir (n bir eksik gorunur -- ornegin HTRU2'de 17898 yerine 17897).
    # Sutun adlari sayisal gorunuyorsa bu "kaybolan" satiri veriye geri ekleyip
    # son sutunu hedef kabul ederek kurtariyoruz (dogrulandi: HTRU2'de bu,
    # tam olarak beklenen 16259/1639 sinif dagilimini geri veriyor).
    if y is None:
        def _looks_numeric(s):
            try:
                float(s)
                return True
            except (TypeError, ValueError):
                return False

        if len(X.columns) > 0 and all(_looks_numeric(c) for c in X.columns):
            print(f"    UYARI: '{openml_name}' icin hedef sutun metadata'si eksik; "
                  f"kaybolan baslik satiri veriye geri ekleniyor, son sutun hedef sayiliyor.")
            header_vals = [float(c) for c in X.columns]
            new_cols = [f'F{i}' for i in range(X.shape[1])]
            header_row = pd.DataFrame([header_vals], columns=new_cols)
            X.columns = new_cols
            X = pd.concat([header_row, X], ignore_index=True)
            y = X.iloc[:, -1]
            X = X.iloc[:, :-1]
        else:
            raise ValueError(
                f"'{openml_name}' icin hedef sutun bulunamadi ve otomatik kurtarma "
                f"uygulanamadi (sutun adlari sayisal gorunmuyor)."
            )

    # Savunma amacli #2: bu veri setleri saf numerik olmali, ama kategori/object
    # kalan bir sutun varsa (beklenmiyor) LabelEncoder ile numerige cevrilir --
    # aksi halde pipeline'in geri kalani (X_df.values.astype(float)) patlar.
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"    UYARI: '{col}' numerik degil, LabelEncoder uygulaniyor.")
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X[col] = X[col].astype(float)

    y_enc = LabelEncoder().fit_transform(pd.Series(y).astype(str))
    df_out = X.copy()
    df_out['target'] = y_enc

    df_out.to_excel(out_path, index=False)
    print(f"    OK: {out_path}  (n={df_out.shape[0]}, p={df_out.shape[1] - 1}, "
          f"pozitif_oran={y_enc.mean():.3f})")


def prepare_all_new_datasets():
    """
    TAMAMEN kutuphane-tabanli/upload-gerektirmeyen hazirlama: 5 veri setinin
    (magic dahil) tumu sklearn.datasets.fetch_openml ile otomatik cekilir --
    hicbir dosya elle yuklenmesi gerekmez. Yalnizca RUN_ONLY'de aktif olan
    veri setleri cekilir (gereksiz indirme/zaman kaybi olmasin diye).

    NOT (magic icin onemli): burada cekilen 'MagicTelescope' (OpenML did=1120),
    UCI MAGIC Gamma Telescope veri setinin HAM hali -- ayni n=19020, ayni 10
    ozellik+hedef. Ama makaledeki magicdata3_READY_FOR_RS.xlsx'e uygulanmis
    olan on-isleme (Winsorization/korelasyon-filtreleme, bkz. makale
    Sinirlamalar bolumu) BU dosyada YOKTUR -- yani buradaki magic sonuclari
    NIHAI_RAPOR.md/makaledeki Tablo ile RAKAM RAKAM kiyaslanamaz, yalnizca
    "gercek n=19020 olceginde V1_tstat/V1_hybrid nasil davraniyor" sorusuna
    BAGIMSIZ bir cevap verir (bu da zaten istenen sey). Makaledeki sayilarla
    birebir kiyaslamak isterseniz, aynen once oldugu gibi kendi
    magicdata3_READY_FOR_RS.xlsx dosyanizi DATA_DIR'e koymaniz yeterli --
    script o dosyayi bulursa OpenML'e hic gitmez (asagidaki "zaten mevcut"
    kontrolu sayesinde), otomatik olarak onu kullanir.
    """
    print("\n" + "=" * 80)
    print("VERI SETLERI HAZIRLANIYOR (OpenML, tek seferlik/idempotent, upload GEREKMEZ)")
    print("=" * 80)
    active_keys = [k for k in OPENML_SPECS if RUN_ONLY is None or k in RUN_ONLY]
    for ds_key in active_keys:
        openml_name = OPENML_SPECS[ds_key]
        filename, _ = DATASETS[ds_key]
        out_path = os.path.join(DATA_DIR, filename)
        try:
            _fetch_and_prepare_openml(ds_key, openml_name, out_path)
        except Exception as e:
            print(f"  HATA [{ds_key}]: OpenML'den cekilemedi ({e!r}) -- "
                  f"bu veri seti run_full_pipeline() sirasinda atlanacak.")


# =============================================================================
# SECTION 2 - YARDIMCI FONKSIYONLAR (TDA_feature_selection_pipeline.py, aynen)
# =============================================================================

def _detect_target(df, hint=None):
    if hint and hint in df.columns:
        return hint
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[-1]


def _is_categorical(series):
    """Obje, bool veya binary integer -> kategorik (binning'e dahil etme)."""
    if series.dtype == object or series.dtype == bool:
        return True
    if pd.api.types.is_integer_dtype(series) and series.nunique() <= 2:
        return True
    return False


def _load_dataset(filename, target_hint=None):
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_excel(path) if filename.endswith(('xlsx', 'xls')) else pd.read_csv(path)

    drop = [c for c in EXCLUDE_COLS if c in df.columns]
    df = df.drop(columns=drop)

    target = _detect_target(df, target_hint)

    if not pd.api.types.is_numeric_dtype(df[target]):
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target].astype(str))

    return df, target


# =============================================================================
# KARISIK-TIP MESAFE (HEOM / overlap metric)
# =============================================================================
# Kategorik (nominal) sutunlar LabelEncoder ile 0,1,2,... kodlanip dogrudan L1'e
# verilirse, kodlarin (alfabetik) sirasi uydurma bir ordinal metrik dayatir:
# |'m'-'s'|=2, |'m'-'n'|=1 gibi. Bu, L1 tolerans iliskisinin anlamini bozar.
# Dogrusu overlap metrigidir: nominal sutunda esitse 0, farkliysa 1
# (Wilson & Martinez 1997, HEOM; Gower 1971).
#
# NOT: 2 seviyeli sutunlarda iki davranis OZDESTIR (kodlar 1 birim arali ise),
# bu yuzden bu duzeltme mevcut 9 veri setinin hicbir sonucunu degistirmez --
# hepsinde nominal sutunlar ya yok ya da ikili (cyron/diabet 'sex', 1/2 kodlu).
# Duzeltme 3+ seviyeli nominal iceren veri setleri icin gereklidir.

CAT_MASK = None      # bool ndarray | None -- sutun bazli "bu sutun nominal mi"


def set_cat_mask(mask):
    """Aktif oznitelik matrisinin nominal sutun maskesini kaydeder."""
    global CAT_MASK
    CAT_MASK = None if mask is None or not np.any(mask) else np.asarray(mask, dtype=bool)


def _mixed_cdist(X, cols=None):
    """L1 (surekli/Jenks kutulari) + overlap (nominal) karisik mesafe matrisi.

    cols : X'in sutunlarinin GLOBAL oznitelik indeksleri (alt kume secildiginde
           maskeyi dogru hizalamak icin). None ise X tum ozniteliklerdir.
    """
    X = np.asarray(X, dtype=float)
    if CAT_MASK is None:
        return cdist(X, X, 'cityblock')
    m = CAT_MASK if cols is None else CAT_MASK[np.asarray(cols, dtype=int)]
    if not np.any(m):
        return cdist(X, X, 'cityblock')
    d = np.zeros((len(X), len(X)), dtype=float)
    if np.any(~m):
        d += cdist(X[:, ~m], X[:, ~m], 'cityblock')
    # 'hamming' farkli koordinat ORANI verir -> sutun sayisiyla carpip sayiya cevir
    ncat = int(m.sum())
    d += cdist(X[:, m], X[:, m], 'hamming') * ncat
    return d


def _get_eps_range(X, n_points=N_EPS_PTS):
    """Veri dagilimina dayali epsilon degerleri (5. - 95. yuzdelik)."""
    d = _mixed_cdist(X)
    upper = d[np.triu_indices(len(X), 1)]
    candidates = np.unique(
        np.round(np.percentile(upper, np.linspace(5, 95, n_points))).astype(int)
    )
    return sorted([int(e) for e in candidates if e > 0])


def _compute_metrics(X, y, eps):
    """Verilen epsilon icin Gamma (alt yaklasim dogrulugu) ve H_labeled hesaplar."""
    n = len(y)
    d_mat = _mixed_cdist(X)
    adj = d_mat <= eps

    adj_no_diag = adj.copy()
    np.fill_diagonal(adj_no_diag, False)

    same_class = (y[:, None] == y[None, :])
    intra = int((adj_no_diag & same_class).sum())
    total = int(adj_no_diag.sum())
    h_labeled = intra / (total + 1e-9)

    adj_with_diag = adj.copy()
    np.fill_diagonal(adj_with_diag, True)

    nr = _neighborhood_closure(adj_with_diag)
    lower_mask = np.all((~nr) | same_class, axis=1)
    gamma = lower_mask.sum() / n

    return float(gamma), float(h_labeled)


def _neighborhood_closure(adj_with_diag):
    """
    nr[i,j] = True iff EVERY tolerance-neighbour k of i (adj_with_diag[i,k])
    is also a neighbour of j (adj_with_diag[k,j]).

    BLAS destekli float32 matris carpimi kullanarak Python dongusu tamamen
    ortadan kaldirilir (dogrulugu birim testle teyit edildi).
    """
    A = adj_with_diag.astype(np.float32)
    not_nr = (A @ (~adj_with_diag).astype(np.float32)) > 0.5
    return ~not_nr


def _compute_gamma_from_dmat(d_mat, y, eps):
    n = len(y)
    adj_with_diag = (d_mat <= eps)
    np.fill_diagonal(adj_with_diag, True)

    same_class = (y[:, None] == y[None, :])
    nr = _neighborhood_closure(adj_with_diag)

    lower_mask = np.all((~nr) | same_class, axis=1)
    return lower_mask.sum() / n


def _compute_h_from_dmat(d_mat, y, eps):
    adj_no_diag = (d_mat <= eps)
    np.fill_diagonal(adj_no_diag, False)
    same_class = (y[:, None] == y[None, :])
    intra = int((adj_no_diag & same_class).sum())
    total = int(adj_no_diag.sum())
    return intra / (total + 1e-9)


def _stratified_subsample(X, y, max_n, random_state=RANDOM_STATE):
    """
    n > max_n ise sinif oranlarini koruyarak alt orneklem indeksi doner.
    Yalnizca O(n^2) topolojik/ReliefF hesaplari icin kullanilir - asagi akis
    siniflandirici egitimi/degerlendirmesi bundan ETKILENMEZ.
    """
    n = len(y)
    if n <= max_n:
        return np.arange(n)
    rng = np.random.RandomState(random_state)
    frac = max_n / n
    idx_parts = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        take = max(1, int(round(len(cls_idx) * frac)))
        take = min(take, len(cls_idx))
        idx_parts.append(rng.choice(cls_idx, size=take, replace=False))
    return np.sort(np.concatenate(idx_parts))


def _quartile_or_positive_select(ranking_df, score_col):
    """
    Tum skorlar pozitifse: ilk ceyrek (Q1) ustu secilir.
    Aksi halde: yalnizca S>0 olanlar secilir.
    """
    scores = ranking_df[score_col].values.astype(float)
    if np.all(scores > 0):
        threshold = float(np.percentile(scores, 25))
        mask = ranking_df[score_col] > threshold
        mode = 'quartile_Q1'
    else:
        threshold = 0.0
        mask = ranking_df[score_col] > 0
        mode = 'strict_positive'

    selected = ranking_df.loc[mask, 'Feature'].tolist()
    if not selected:
        selected = ranking_df.sort_values(score_col, ascending=False)['Feature'].iloc[:1].tolist()
    return selected, threshold, mode


def _median_select(ranking_df, score_col):
    """Q1 kuralina duyarlilik karsilastirmasi icin: medyan-ustu esik."""
    threshold = float(ranking_df[score_col].median())
    mask = ranking_df[score_col] > threshold
    selected = ranking_df.loc[mask, 'Feature'].tolist()
    if not selected:
        selected = ranking_df.sort_values(score_col, ascending=False)['Feature'].iloc[:1].tolist()
    return selected, threshold, 'median'


# =============================================================================
# SECTION 2b - Faz 2 adaylari (V1_tstat, V1_floor30, V1_hybrid) - aynen
# =============================================================================

def _tstat_select(ranking_df, score_col='Delta_Gamma', sem_col='SEM_Gamma', kappa=1.5):
    """
    Kararlilik-farkindali secim kurali (eq:v3_selection, explain-rough-only.tex).
    Q1'in ustunde olmak yetmez; esigi en az kappa standart hata (SEM_f =
    sigma_f/sqrt(N)) kadar asmasi gerekir. Kural Stability'ye degil
    Delta_Gamma'ya uygulanir (Stability zaten bir S/N orani).
    """
    scores = ranking_df[score_col].values.astype(float)
    sems = ranking_df[sem_col].values.astype(float)
    q1 = float(np.percentile(scores, 25))
    t = (scores - q1) / (sems + 1e-9)
    mask = t > kappa
    selected = ranking_df.loc[mask, 'Feature'].tolist()
    if not selected:
        selected = ranking_df.sort_values(score_col, ascending=False)['Feature'].iloc[:1].tolist()
    return selected, q1, f'tstat_kappa{kappa}'


def _fixed_floor_select(ranking_df, score_col, floor_fraction, p_total):
    """Mevcut Q1/strict-positive kurali + tek bir a-priori taban (floor_fraction)."""
    selected, threshold, mode = _quartile_or_positive_select(ranking_df, score_col)
    floor_k = max(2, math.ceil(floor_fraction * p_total))
    if len(selected) < floor_k:
        selected = ranking_df.sort_values(score_col, ascending=False)['Feature'].iloc[:floor_k].tolist()
        mode = mode + '+floor'
    return selected, threshold, mode


def _hybrid_select(v1_ranking, cls_ranking, score_col='Stability'):
    """
    Gamma-PFI (topolojik) skorunu klasik yontem konsensusuyla (Classical_Mean_Rank)
    birlestiren hibrit siralama. Her ikisi de z-skorlanip toplanir, Q1 kurali
    birlesik skora uygulanir.
    """
    merged = v1_ranking[['Feature', score_col]].merge(
        cls_ranking[['Feature', 'Classical_Mean_Rank']], on='Feature'
    )
    topo = merged[score_col].astype(float)
    cls_rank = merged['Classical_Mean_Rank'].astype(float)
    z_topo = (topo - topo.mean()) / (topo.std(ddof=0) + 1e-9)
    z_cls = -(cls_rank - cls_rank.mean()) / (cls_rank.std(ddof=0) + 1e-9)
    merged = merged.copy()
    merged['Hybrid_Score'] = z_topo + z_cls
    selected, threshold, mode = _quartile_or_positive_select(merged, 'Hybrid_Score')
    return selected, threshold, mode


def _gamma_retention(X_sub, features, selected, y_sub, eps):
    """
    Gozetimsiz alt-kume kalite skoru: secilen ozellik alt kumesinin KENDI
    L1/tolerans yapisiyla hesaplanan gamma dogrulugu, tam ozellik uzayindaki
    gamma_base'e orani.
    """
    idx = [features.index(f) for f in selected if f in features]
    if not idx:
        return np.nan
    X_selected = X_sub[:, idx]
    d_sub = _mixed_cdist(X_selected, cols=idx)
    return _compute_gamma_from_dmat(d_sub, y_sub, eps)


# =============================================================================
# SECTION 3 - AutoJenksDiscretizer (aynen)
# =============================================================================

class AutoJenksDiscretizer:
    """
    KMeans tabanli Jenks binning. GVF hedefine ulasana kadar k artirir.
    fit() SADECE train verisiyle cagrilir; transform() bin merkezlerini
    train'den miras alarak train VE test'e ayni sekilde uygulanir.
    """

    def __init__(self, target_gvf=JENKS_GVF, max_bins=JENKS_BINS, random_state=RANDOM_STATE):
        self.target_gvf = target_gvf
        self.max_bins = max_bins
        self.random_state = random_state
        self.column_mode_ = {}
        self.column_centers_ = {}
        self.column_k_ = {}
        self.column_gvf_ = {}

    @staticmethod
    def _gvf(X_flat, labels, centers):
        mean = X_flat.mean()
        sdam = ((X_flat - mean) ** 2).sum()
        if sdam == 0:
            return 1.0
        sdcm = sum(((X_flat[labels == i] - c) ** 2).sum() for i, c in enumerate(centers))
        return (sdam - sdcm) / sdam

    def fit(self, X_df):
        for col in X_df.columns:
            ser = X_df[col]
            if _is_categorical(ser):
                self.column_mode_[col] = 'categorical'
                self.column_k_[col] = int(ser.nunique())
                continue

            X_col = ser.values.reshape(-1, 1).astype(float)
            unique_vals = np.unique(X_col)
            if len(unique_vals) <= 2:
                self.column_mode_[col] = 'categorical'
                self.column_k_[col] = len(unique_vals)
                continue

            best_centers, optimal_k, best_gvf = unique_vals[:1], 1, 1.0
            for k in range(2, min(self.max_bins + 1, len(unique_vals))):
                # PERFORMANS: n_init=5 orijinaldeki gibi korundu (sonucu
                # etkileyen bir hiperparametre, degistirilmedi).
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
                lbls = km.fit_predict(X_col)
                ctrs = km.cluster_centers_.flatten()
                gvf = self._gvf(X_col.flatten(), lbls, ctrs)
                best_centers, optimal_k, best_gvf = ctrs, k, gvf
                if gvf >= self.target_gvf:
                    break

            self.column_mode_[col] = 'continuous'
            self.column_centers_[col] = np.sort(best_centers)
            self.column_k_[col] = optimal_k
            self.column_gvf_[col] = float(best_gvf)
        return self

    def transform(self, X_df):
        res = X_df.copy()
        for col in res.columns:
            if self.column_mode_.get(col) != 'continuous':
                continue
            centers = self.column_centers_[col]
            vals = res[col].values.reshape(-1, 1).astype(float)
            dists = np.abs(vals - centers.reshape(1, -1))
            labels = dists.argmin(axis=1)
            res[col] = labels + 1
        return res

    def fit_transform(self, X_df):
        return self.fit(X_df).transform(X_df)


# =============================================================================
# SECTION 4 - V1Selector (H_labeled'siz) - aynen
# =============================================================================

class V1Selector:
    """
    Kaba Kume tabanli ozellik secimi - H_labeled kullanmaz.
    Epsilon secimi: argmax(Gamma) kosuluyla Gamma < 0.999.
    PFI skoru: Delta_Gamma. Kararlilik: mean(Delta_Gamma)/std(Delta_Gamma).
    """

    def __init__(self, target_col, n_rep=N_REP, random_state=RANDOM_STATE):
        self.target_col = target_col
        self.n_rep = n_rep
        self.random_state = random_state
        self.eps_star = None
        self.gamma_base = None
        self.eps_scan_ = None
        self.ranking_ = None
        self.n_topological_ = None
        self.features_ = None
        self.X_sub_ = None
        self.y_sub_ = None

    def fit(self, df_disc_train):
        features = [c for c in df_disc_train.columns if c != self.target_col]
        X_full = df_disc_train[features].values.astype(float)
        y_full = df_disc_train[self.target_col].values.astype(float)

        sub_idx = _stratified_subsample(X_full, y_full, MAX_N_TOPOLOGICAL, self.random_state)
        X, y = X_full[sub_idx], y_full[sub_idx]
        self.n_topological_ = len(sub_idx)
        self.features_ = features
        self.X_sub_ = X
        self.y_sub_ = y

        eps_range = _get_eps_range(X)
        rows = []
        for eps in eps_range:
            gamma, _ = _compute_metrics(X, y, eps)
            rows.append({'eps': eps, 'gamma': gamma})
        res = pd.DataFrame(rows)
        valid = res[res['gamma'] < 0.999]
        best = (valid if not valid.empty else res)
        best = best.loc[best['gamma'].idxmax()]
        self.eps_star = float(best['eps'])
        self.gamma_base = float(best['gamma'])
        self.eps_scan_ = res

        rng = np.random.RandomState(self.random_state)
        d_base = _mixed_cdist(X)
        # BELLEK: eskiden p adet n x n matris onceden hesaplanip TUMU bellekte
        # tutuluyordu (p * n^2 * 8 bayt -- p=10935, n=1545 icin 209 GB). Her matris
        # yalnizca kendi ozniteliginin dongusunde kullanildigi icin dongu icine
        # alindi: bellek O(p*n^2) -> O(n^2). Zaman bedeli n_rep=30'da ~%2.
        scores = []
        for i, col in enumerate(features):
            col_dist_i = np.abs(X[:, i][:, None] - X[:, i][None, :])
            delta_list = []
            for _ in range(self.n_rep):
                perm = rng.permutation(X[:, i])
                d_perm = d_base - col_dist_i + np.abs(perm[:, None] - perm[None, :])
                g_perm = _compute_gamma_from_dmat(d_perm, y, self.eps_star)
                delta_list.append(self.gamma_base - g_perm)
            del col_dist_i
            mean_dg = float(np.mean(delta_list))
            std_dg = float(np.std(delta_list))
            stability = mean_dg / (std_dg + 1e-9)
            sem_dg = std_dg / np.sqrt(self.n_rep)
            t_stat = abs(mean_dg) * np.sqrt(self.n_rep) / (std_dg + 1e-9)
            scores.append({
                'Feature': col, 'Delta_Gamma': mean_dg, 'Std_Gamma': std_dg,
                'SEM_Gamma': sem_dg, 'T_stat': t_stat, 'Stability': stability,
            })

        df_scores = pd.DataFrame(scores).sort_values('Delta_Gamma', ascending=False).reset_index(drop=True)
        df_scores['V1_Rank'] = df_scores.index + 1
        self.ranking_ = df_scores
        return self

    def select_final(self, score_col='Stability', rule='quartile'):
        if rule == 'median':
            return _median_select(self.ranking_, score_col)
        return _quartile_or_positive_select(self.ranking_, score_col)


# =============================================================================
# SECTION 4b - V1WeightedSelector (Faz 3 - negatif sonuc, karsilastirma icin) - aynen
# =============================================================================

class V1WeightedSelector(V1Selector):
    """
    V1Selector ile ayni, ama L1 tolerans mesafesi hesaplanmadan once her
    ozellik sutunu ucuz bir ilgi-duzeyi proxy'siyle (ANOVA F-istatistigi,
    yalniz train fold'da hesaplanir) agirliklandiriliyor.
    """

    def fit(self, df_disc_train):
        features = [c for c in df_disc_train.columns if c != self.target_col]
        X = df_disc_train[features].values.astype(float)
        y = df_disc_train[self.target_col].values.astype(int)

        f_vals, _ = f_classif(X, y)
        f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
        weights = (f_vals + 1e-6) / (f_vals.mean() + 1e-6)
        self.feature_weights_ = dict(zip(features, weights))

        df_scaled = df_disc_train.copy()
        for f, w in zip(features, weights):
            df_scaled[f] = df_scaled[f] * w

        return super().fit(df_scaled)


# =============================================================================
# SECTION 5 - V2Selector (H_labeled dahil) - aynen
# =============================================================================

class V2Selector:
    """
    Kaba Kume + H_labeled tabanli ozellik secimi.
    Epsilon secimi: joint(Gamma x H_labeled) skorunun yerel maksimumu,
    yoksa global maksimum. PFI skoru: Delta_Gamma + Delta_H.
    """

    def __init__(self, target_col, n_rep=N_REP, random_state=RANDOM_STATE):
        self.target_col = target_col
        self.n_rep = n_rep
        self.random_state = random_state
        self.eps_star = None
        self.gamma_base = None
        self.h_base = None
        self.score_mode = None
        self.eps_scan_ = None
        self.ranking_ = None
        self.n_topological_ = None

    def fit(self, df_disc_train):
        features = [c for c in df_disc_train.columns if c != self.target_col]
        X_full = df_disc_train[features].values.astype(float)
        y_full = df_disc_train[self.target_col].values.astype(float)

        sub_idx = _stratified_subsample(X_full, y_full, MAX_N_TOPOLOGICAL, self.random_state)
        X, y = X_full[sub_idx], y_full[sub_idx]
        self.n_topological_ = len(sub_idx)

        eps_range = _get_eps_range(X)
        rows = []
        for eps in eps_range:
            gamma, h = _compute_metrics(X, y, eps)
            rows.append({'eps': eps, 'gamma': gamma, 'h_labeled': h})
        res = pd.DataFrame(rows)
        valid = res[res['gamma'] < 0.999].reset_index(drop=True)

        max_g = res['gamma'].max() + 1e-9
        max_h = res['h_labeled'].max() + 1e-9

        if not valid.empty:
            valid = valid.copy()
            valid['joint'] = (valid['gamma'] / max_g) * (valid['h_labeled'] / max_h)
            local_max = [
                valid.iloc[i] for i in range(1, len(valid) - 1)
                if valid['joint'].iloc[i] > valid['joint'].iloc[i - 1]
                and valid['joint'].iloc[i] >= valid['joint'].iloc[i + 1]
            ]
            best = max(local_max, key=lambda r: r['joint']) if local_max else valid.loc[valid['joint'].idxmax()]
        else:
            res['joint'] = (res['gamma'] / max_g) * (res['h_labeled'] / max_h)
            best = res.loc[res['joint'].idxmax()]

        self.eps_star = float(best['eps'])
        self.gamma_base = float(best['gamma'])
        self.h_base = float(best['h_labeled'])
        self.eps_scan_ = res

        rng = np.random.RandomState(self.random_state)
        d_base = _mixed_cdist(X)
        # BELLEK: bkz. V1Selector -- col_dists on-bellege alinmiyor, dongu icinde.
        raw = []
        all_dg, all_dh = [], []
        for i, col in enumerate(features):
            col_dist_i = np.abs(X[:, i][:, None] - X[:, i][None, :])
            dg_list, dh_list = [], []
            for _ in range(self.n_rep):
                perm = rng.permutation(X[:, i])
                d_perm = d_base - col_dist_i + np.abs(perm[:, None] - perm[None, :])
                g_perm = _compute_gamma_from_dmat(d_perm, y, self.eps_star)
                h_perm = _compute_h_from_dmat(d_perm, y, self.eps_star)
                dg_list.append(self.gamma_base - g_perm)
                dh_list.append(self.h_base - h_perm)
            del col_dist_i
            mean_dg, mean_dh = float(np.mean(dg_list)), float(np.mean(dh_list))
            std_dg = float(np.std(dg_list))
            stability = mean_dg / (std_dg + 1e-9)
            raw.append({'Feature': col, 'Delta_Gamma': mean_dg, 'Delta_H': mean_dh, 'Stability': stability})
            all_dg.append(mean_dg)
            all_dh.append(mean_dh)

        df_scores = pd.DataFrame(raw)
        corr = float(np.corrcoef(all_dg, all_dh)[0, 1]) if len(set(all_dg)) > 1 else 0.0
        max_dg = max(abs(v) for v in all_dg) + 1e-9
        max_dh = max(abs(v) for v in all_dh) + 1e-9

        if abs(corr) > 0.8:
            df_scores['Final_Score'] = df_scores['Delta_Gamma'] / max_dg + df_scores['Delta_H'] / max_dh
            self.score_mode = 'Additive'
        else:
            df_scores['Final_Score'] = (
                df_scores['Delta_Gamma'].clip(lower=0) / max_dg *
                df_scores['Delta_H'].clip(lower=0) / max_dh
            )
            self.score_mode = 'Multiplicative'

        df_scores = df_scores.sort_values('Final_Score', ascending=False).reset_index(drop=True)
        df_scores['V2_Rank'] = df_scores.index + 1
        self.ranking_ = df_scores
        return self

    def select_final(self, score_col='Stability', rule='quartile'):
        if rule == 'median':
            return _median_select(self.ranking_, score_col)
        return _quartile_or_positive_select(self.ranking_, score_col)


# =============================================================================
# SECTION 6 - ClassicalMethods (+ ReliefF, Standard PFI) - PERFORMANS: n_jobs=-1
# eklendi (RF/RFE/permutation_importance icin) -- sonuclari ETKILEMEZ, yalnizca
# Colab'in coklu-cekirdegini kullanarak hizlandirir.
# =============================================================================

class ClassicalMethods:
    """Ham (binlenmemis) TRAIN verisi uzerinde 9 klasik ozellik secim yontemi."""

    def __init__(self, target_col, random_state=RANDOM_STATE):
        self.target_col = target_col
        self.random_state = random_state

    def run(self, X_df, y):
        features = list(X_df.columns)
        X = X_df.values.astype(float)
        y = np.asarray(y).astype(int)

        scores = {'Feature': features}

        f_vals, _ = f_classif(X, y)
        scores['ANOVA_Score'] = f_vals

        mi = mutual_info_classif(X, y, random_state=self.random_state)
        scores['MI_Score'] = mi

        lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=self.random_state)
        lasso.fit(X, y)
        coef = lasso.coef_
        scores['LASSO_Score'] = np.abs(coef).max(axis=0) if coef.ndim > 1 else np.abs(coef)

        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf.fit(X, y)
        scores['RF_Score'] = rf.feature_importances_

        gb = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
        gb.fit(X, y)
        scores['GB_Score'] = gb.feature_importances_

        rfe_rf = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        rfe = RFE(rfe_rf, n_features_to_select=1)
        rfe.fit(X, y)
        scores['RFE_Score'] = (len(features) + 1 - rfe.ranking_).astype(float)

        shap_rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        shap_rf.fit(X, y)
        explainer = shap.TreeExplainer(shap_rf)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_imp = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
        elif shap_vals.ndim == 3:
            shap_imp = np.abs(shap_vals).mean(axis=(0, 2))
        else:
            shap_imp = np.abs(shap_vals).mean(axis=0)
        scores['SHAP_Score'] = shap_imp

        # --- ReliefF (O(n^2)-ish komsuluk arama; buyuk n icin alt orneklenir) ---
        sub_idx = _stratified_subsample(X, y, MAX_N_TOPOLOGICAL, self.random_state)
        X_rel, y_rel = X[sub_idx], y[sub_idx]
        n_neighbors = max(3, min(RELIEF_N_NEIGHBORS, len(y_rel) - 1))
        relief = ReliefF(n_neighbors=n_neighbors)
        relief.fit(X_rel, y_rel)
        scores['ReliefF_Score'] = relief.feature_importances_

        # --- Standart (Gamma'siz) Permutation Importance ---
        pi_rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        pi_rf.fit(X, y)
        pi = permutation_importance(
            pi_rf, X, y, n_repeats=N_REP_STD_PFI,
            random_state=self.random_state, scoring='accuracy', n_jobs=-1,
        )
        scores['StdPFI_Score'] = pi.importances_mean

        df = pd.DataFrame(scores)
        for sc in CLASSICAL_SCORE_COLS:
            rank_col = sc.replace('_Score', '_Rank')
            df[rank_col] = df[sc].rank(ascending=False, method='min').astype(int)

        rank_cols = [c for c in df.columns if c.endswith('_Rank')]
        df['Classical_Mean_Rank'] = df[rank_cols].mean(axis=1)
        df = df.sort_values('Classical_Mean_Rank').reset_index(drop=True)
        return df


# =============================================================================
# SECTION 6b - yz_algorithms.py'den YALNIZCA fiilen kullanilan zincir
# (train_models_given_split ve bagimliliklari; CONFIG/Universal_YZ_With_Config/
# run_experiment vb. legacy/kullanilmayan kisimlar bilerek atlandi)
# =============================================================================

def calculate_metrics(y_true, y_pred, y_score=None):
    num_classes = len(np.unique(y_true))
    avg_method = 'binary' if num_classes == 2 else 'macro'

    metrics = {
        'ACCU': accuracy_score(y_true, y_pred),
        'PREC': precision_score(y_true, y_pred, average=avg_method, zero_division=0),
        'RECA': recall_score(y_true, y_pred, average=avg_method, zero_division=0),
        'F1_S': f1_score(y_true, y_pred, average=avg_method, zero_division=0),
    }

    if y_score is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_score)
        except ValueError:
            metrics['AUC'] = np.nan
    else:
        metrics['AUC'] = np.nan

    return metrics


def build_base_models(random_state=42, xgb_scale_weight=1):
    """
    Bkz. makale Ek Tablo "Classifier Hyperparameters" -- TUM hiperparametreler
    aynen korundu. PERFORMANS: RFC/XGBC'ye n_jobs=-1 eklendi (sonucu etkilemez).
    """
    return {
        'GBC': GradientBoostingClassifier(random_state=random_state),
        'GNB': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'LOGREG': LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced'),
        'RFC': RandomForestClassifier(random_state=random_state, class_weight='balanced', n_jobs=-1),
        'SVC': svm.SVC(kernel='rbf', C=1.0, random_state=random_state, class_weight='balanced'),
        'XGBC': XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            random_state=random_state,
            scale_pos_weight=xgb_scale_weight,
            n_jobs=-1,
        ),
    }


SCALED_MODELS = ('GBC', 'SVC', 'LOGREG', 'LDA')


def train_models_given_split(X_train, X_test, y_train, y_test, random_state=42):
    """
    HYBRID modeli: 7 temel siniflandiricinin (GBC, GNB, LDA, LOGREG, RFC, SVC,
    XGBC) test seti tahminleri uzerinden yapilan SERT COGUNLUK OYLAMASIDIR.
    """
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    xgb_scale_weight = (num_neg / num_pos) if num_pos > 0 else 1

    models = build_base_models(random_state=random_state, xgb_scale_weight=xgb_scale_weight)

    results = {}
    preds_pool = []

    for name, model in models.items():
        Xtr, Xte = (X_train_s, X_test_s) if name in SCALED_MODELS else (X_train, X_test)
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)

        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(Xte)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_score = model.decision_function(Xte)
        else:
            y_score = preds

        results[name] = calculate_metrics(y_test, preds, y_score)
        preds_pool.append(preds)

    preds_pool = np.array(preds_pool)
    sum_votes = np.sum(preds_pool, axis=0)
    threshold = len(models) / 2
    hybrid_preds = np.array([1 if vote > threshold else 0 for vote in sum_votes])
    hybrid_score = sum_votes / len(models)

    results['HYBRID'] = calculate_metrics(y_test, hybrid_preds, hybrid_score)

    return results


# =============================================================================
# SECTION 7 - Siniflandirici degerlendirmesi yardimcilari (aynen)
# =============================================================================

EVAL_CLASSIFIERS = ['GBC', 'GNB', 'LDA', 'LOGREG', 'RFC', 'SVC', 'XGBC', 'HYBRID']


def _cap_train_for_classifier(X_train, y_train, max_n=MAX_N_CLASSIFIER_TRAIN, random_state=RANDOM_STATE):
    """SVC (rbf) gibi O(n^2-n^3) maliyetli modeller icin train boyutu ustsiniri."""
    if len(y_train) <= max_n:
        return X_train, y_train
    idx = _stratified_subsample(X_train, y_train, max_n, random_state)
    return X_train[idx], y_train[idx]


def _evaluate_feature_set(X_train_df, X_test_df, y_train, y_test, feature_list, random_state=RANDOM_STATE):
    cols = [f for f in feature_list if f in X_train_df.columns]
    if not cols:
        cols = list(X_train_df.columns)
    Xtr = X_train_df[cols].values.astype(float)
    Xte = X_test_df[cols].values.astype(float)
    Xtr_c, ytr_c = _cap_train_for_classifier(Xtr, y_train, random_state=random_state)
    return train_models_given_split(Xtr_c, Xte, ytr_c, y_test, random_state=random_state), len(cols)


# =============================================================================
# SECTION 8 - Nested-CV ana dongusu (aynen)
# =============================================================================

def run_dataset(ds_name, filename, target_hint=None,
                 n_splits=N_SPLITS_OUTER, n_repeats=N_REPEATS_OUTER):
    print("\n" + "=" * 80)
    print(f"DATASET: {ds_name.upper()}  (nested CV: {n_splits} fold x {n_repeats} tekrar)")
    print("=" * 80)

    df_raw, target = _load_dataset(filename, target_hint)
    features_all = [c for c in df_raw.columns if c != target]
    X_all_df = df_raw[features_all]
    y_all = df_raw[target].values.astype(int)
    print(f"  Sekil: {df_raw.shape} | Hedef: '{target}' | Ozellik sayisi: {len(features_all)}")

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

    fold_rows = []
    epsq1_rows = []
    corr_rows = []
    ranking_detail_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X_all_df, y_all), start=1):
        fold_t0 = time.time()
        X_train_df = X_all_df.iloc[train_idx].reset_index(drop=True)
        X_test_df = X_all_df.iloc[test_idx].reset_index(drop=True)
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        disc = AutoJenksDiscretizer(random_state=RANDOM_STATE)
        disc.fit(X_train_df)
        X_train_disc = disc.transform(X_train_df)
        # Nominal sutunlar L1 yerine overlap metrigiyle olculur (bkz. _mixed_cdist).
        # Maske oznitelik sirasina gore kurulur; V1Selector'un kullandigi sira da bu.
        feat_order = [c for c in X_train_disc.columns]
        set_cat_mask([disc.column_mode_.get(c) == 'categorical'
                      for c in feat_order])
        df_disc_train = pd.concat(
            [X_train_disc, pd.Series(y_train, name=target)], axis=1
        )

        v1 = V1Selector(target, random_state=RANDOM_STATE).fit(df_disc_train)
        v2 = None   # LEAN: makale V2'yi raporlamiyor, pahali fit atlandi
        v1w = None  # LEAN: makale V1_weighted'i raporlamiyor

        v1_final, v1_thr, v1_mode = v1.select_final('Stability', rule='quartile')
        v2_final, v2_thr, v2_mode = [], float('nan'), 'NA'
        v1_final_med, v1_thr_med, _ = v1.select_final('Stability', rule='median')
        v2_final_med, v2_thr_med = [], float('nan')
        v1w_final, v1w_thr, v1w_mode = [], float('nan'), 'NA'

        cls_ranking = ClassicalMethods(target, random_state=RANDOM_STATE).run(X_train_df, y_train)

        p_total_features = len(features_all)
        v1_tstat_final, _, v1_tstat_mode = _tstat_select(
            v1.ranking_, 'Delta_Gamma', 'SEM_Gamma', kappa=1.5)
        v1_floor_final, _, v1_floor_mode = _fixed_floor_select(
            v1.ranking_, 'Stability', floor_fraction=0.3, p_total=p_total_features)
        v1_hybrid_final, _, v1_hybrid_mode = _hybrid_select(
            v1.ranking_, cls_ranking, 'Stability')

        for _, r in v1.ranking_.iterrows():
            confidently_negative = bool(r['Delta_Gamma'] < 0 and r['T_stat'] > 1.5)
            ranking_detail_rows.append({
                'Dataset': ds_name, 'Fold': fold_id, 'Feature': r['Feature'],
                'Delta_Gamma': r['Delta_Gamma'], 'Std_Gamma': r['Std_Gamma'],
                'SEM_Gamma': r['SEM_Gamma'], 'T_stat': r['T_stat'],
                'Stability': r['Stability'], 'V1_Rank': r['V1_Rank'],
                'Confidently_Negative': confidently_negative,
            })

        gamma_retention = {}
        for _name, _feats in [
            ('V1_final', v1_final), ('V1_tstat', v1_tstat_final),
        ]:
            g_sub = _gamma_retention(v1.X_sub_, v1.features_, _feats, v1.y_sub_, v1.eps_star)
            gamma_retention[_name] = g_sub / (v1.gamma_base + 1e-9) if v1.gamma_base else np.nan

        epsq1_rows.append({
            'Dataset': ds_name, 'Fold': fold_id,
            'V1_eps_star': v1.eps_star, 'V1_gamma_base': v1.gamma_base,
            'V1_n_topological': v1.n_topological_,
            'V1_Q1_threshold': v1_thr, 'V1_Q1_mode': v1_mode, 'V1_n_selected_Q1': len(v1_final),
            'V1_median_threshold': v1_thr_med, 'V1_n_selected_median': len(v1_final_med),
            'V2_eps_star': None, 'V2_gamma_base': None, 'V2_h_base': None,
            'V2_score_mode': None,
            'V2_Q1_threshold': v2_thr, 'V2_Q1_mode': v2_mode, 'V2_n_selected_Q1': len(v2_final),
            'V2_median_threshold': v2_thr_med, 'V2_n_selected_median': len(v2_final_med),
            'N_features_total': len(features_all),
            'V1_final_GammaRetention': gamma_retention.get('V1_final'),
            'V1_tstat_GammaRetention': gamma_retention.get('V1_tstat'),
            'V1_floor30_GammaRetention': gamma_retention.get('V1_floor30'),
            'V1_hybrid_GammaRetention': gamma_retention.get('V1_hybrid'),
        })

        merged = v1.ranking_[['Feature', 'V1_Rank']].merge(cls_ranking, on='Feature')
        merged['V2_Rank'] = np.nan
        for rank_col in [c for c in merged.columns if c.endswith('_Rank') and c not in ('V1_Rank', 'V2_Rank')]:
            method_name = rank_col.replace('_Rank', '')
            rho_v1, _ = spearmanr(merged['V1_Rank'], merged[rank_col])
            rho_v2 = np.nan
            corr_rows.append({
                'Dataset': ds_name, 'Fold': fold_id, 'Classical_Method': method_name,
                'Spearman_V1_vs_Classical': rho_v1, 'Spearman_V2_vs_Classical': rho_v2,
            })

        # DUZELTME 2026-07-22: matched-K, makalenin raporladigi yontem olan
        # CASPER-t (V1_tstat) kardinalitesine eslestirilir. Onceki surumde
        # V1_final'e eslestiriliyordu; V1_final MAGIC/HTRU2'de K=1'e coktugu
        # icin klasik temeller haksiz sekilde 1 ozellikle kosuluyordu.
        k_match = max(1, len(v1_tstat_final))
        method_feature_sets = {
            'ALL': features_all,
            'V1_tstat': v1_tstat_final,
        }
        for sc in CLASSICAL_SCORE_COLS:
            rank_col = sc.replace('_Score', '_Rank')
            method_name = sc.replace('_Score', '')
            ordered = cls_ranking.sort_values(rank_col)['Feature'].tolist()
            method_feature_sets[f'{method_name}_matchK'] = ordered[:k_match]

        for method_name, feat_list in method_feature_sets.items():
            results, n_used = _evaluate_feature_set(
                X_train_df, X_test_df, y_train, y_test, feat_list, random_state=RANDOM_STATE
            )
            for clf_name, metrics in results.items():
                fold_rows.append({
                    'Dataset': ds_name, 'Fold': fold_id, 'Method': method_name,
                    'N_Features_Used': n_used, 'Classifier': clf_name,
                    **metrics,
                })

        fold_dt = time.time() - fold_t0
        print(f"  [Fold {fold_id}/{n_splits * n_repeats}] ({fold_dt:.0f}s) "
              f"V1: eps*={v1.eps_star} K={len(v1_final)} ({v1_mode}) | "
              f"tstat: K={len(v1_tstat_final)} ({v1_tstat_mode}) | "
              f"floor30: K={len(v1_floor_final)} ({v1_floor_mode}) | "
              f"hybrid: K={len(v1_hybrid_final)} ({v1_hybrid_mode}) | "
              f"weighted: K={len(v1w_final)} ({v1w_mode})")

    fold_df = pd.DataFrame(fold_rows)
    epsq1_df = pd.DataFrame(epsq1_rows)
    corr_df = pd.DataFrame(corr_rows)
    ranking_detail_df = pd.DataFrame(ranking_detail_rows)

    METRIC_COLS = ['ACCU', 'PREC', 'RECA', 'F1_S', 'AUC']
    summary_df = (
        fold_df.groupby(['Method', 'Classifier'])[METRIC_COLS]
        .agg(['mean', 'std']).reset_index()
    )
    summary_df.columns = ['_'.join(c).strip('_') for c in summary_df.columns]
    summary_df.insert(0, 'Dataset', ds_name)

    sig_rows = []
    for clf_name in EVAL_CLASSIFIERS:
        for metric in ['ACCU', 'F1_S', 'AUC']:
            all_vals = (fold_df[(fold_df.Method == 'ALL') & (fold_df.Classifier == clf_name)]
                        .sort_values('Fold')[metric].values)
            for method_name in ['V1_tstat']:
                sel_vals = (fold_df[(fold_df.Method == method_name) & (fold_df.Classifier == clf_name)]
                            .sort_values('Fold')[metric].values)
                valid = (len(all_vals) == len(sel_vals) and len(all_vals) >= 6
                         and not np.any(np.isnan(all_vals)) and not np.any(np.isnan(sel_vals))
                         and np.any(all_vals != sel_vals))
                if valid:
                    try:
                        stat, p = wilcoxon(sel_vals, all_vals)
                    except ValueError:
                        stat, p = np.nan, np.nan
                else:
                    stat, p = np.nan, np.nan
                sig_rows.append({
                    'Dataset': ds_name, 'Classifier': clf_name, 'Metric': metric,
                    'Comparison': f'{method_name}_vs_ALL',
                    'Mean_Selected': float(np.nanmean(sel_vals)) if len(sel_vals) else np.nan,
                    'Mean_ALL': float(np.nanmean(all_vals)) if len(all_vals) else np.nan,
                    'Wilcoxon_stat': stat, 'p_value': p, 'N_folds': len(all_vals),
                })
    sig_df = pd.DataFrame(sig_rows)

    os.makedirs(PER_DS_DIR, exist_ok=True)
    out_path = os.path.join(PER_DS_DIR, f"{ds_name}_TopologicalResults.xlsx")
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        fold_df.to_excel(writer, sheet_name='Fold_Detail', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        sig_df.to_excel(writer, sheet_name='Significance', index=False)
        corr_df.to_excel(writer, sheet_name='Rank_Correlation', index=False)
        epsq1_df.to_excel(writer, sheet_name='Epsilon_Q1_Log', index=False)
        ranking_detail_df.to_excel(writer, sheet_name='V1_Ranking_Detail', index=False)
    print(f"\n  OK {out_path} yazildi.")

    return {
        'fold': fold_df, 'summary': summary_df, 'sig': sig_df,
        'corr': corr_df, 'epsq1': epsq1_df, 'ranking_detail': ranking_detail_df,
    }


def _method_list():
    classical = [sc.replace('_Score', '_matchK') for sc in CLASSICAL_SCORE_COLS]
    return ['ALL', 'V1_tstat'] + classical


def _metric_matrix(master_summary, metric_col, classifier_selector):
    methods = _method_list()
    df = master_summary[master_summary['Method'].isin(methods)]
    if classifier_selector == 'MEAN_OF_CLASSIFIERS':
        df = df[df['Classifier'].isin(EVAL_CLASSIFIERS)]
        piv = df.groupby(['Dataset', 'Method'])[metric_col].mean().reset_index()
    else:
        df = df[df['Classifier'] == classifier_selector]
        piv = df[['Dataset', 'Method', metric_col]]
    mat = piv.pivot(index='Dataset', columns='Method', values=metric_col)
    mat = mat[[m for m in methods if m in mat.columns]]
    return mat.dropna(axis=0, how='any')


def run_friedman_nemenyi(master_summary):
    """Coklu-yontem x coklu-veri-seti Friedman + (anlamliysa) Nemenyi post-hoc testi."""
    rows_summary, rows_ranks, rows_nemenyi = [], [], []
    metrics = [('ACCU_mean', 'Accuracy'), ('F1_S_mean', 'F1'), ('AUC_mean', 'AUC')]
    selectors = ['HYBRID', 'MEAN_OF_CLASSIFIERS']

    for metric_col, metric_name in metrics:
        for sel in selectors:
            mat = _metric_matrix(master_summary, metric_col, sel)
            if mat.shape[0] < 3 or mat.shape[1] < 3:
                rows_summary.append({
                    'Metric': metric_name, 'ClassifierSelection': sel,
                    'N_Datasets': mat.shape[0], 'N_Methods': mat.shape[1],
                    'Friedman_stat': np.nan, 'Friedman_p': np.nan,
                    'Note': 'Yetersiz veri (dataset veya method sayisi < 3)',
                })
                continue

            stat, p = friedmanchisquare(*[mat[c].values for c in mat.columns])
            rows_summary.append({
                'Metric': metric_name, 'ClassifierSelection': sel,
                'N_Datasets': mat.shape[0], 'N_Methods': mat.shape[1],
                'Friedman_stat': float(stat), 'Friedman_p': float(p),
                'Note': '',
            })

            ranks = (-mat).rank(axis=1, method='average')
            mean_ranks = ranks.mean(axis=0).sort_values()
            for method_name, r in mean_ranks.items():
                rows_ranks.append({
                    'Metric': metric_name, 'ClassifierSelection': sel,
                    'Method': method_name, 'Mean_Rank': float(r),
                })

            if p < 0.05:
                nem = sp.posthoc_nemenyi_friedman(mat.values)
                nem.index = mat.columns
                nem.columns = mat.columns
                for a in nem.index:
                    for b in nem.columns:
                        if a < b:
                            rows_nemenyi.append({
                                'Metric': metric_name, 'ClassifierSelection': sel,
                                'Method_A': a, 'Method_B': b,
                                'Nemenyi_p': float(nem.loc[a, b]),
                            })

    return (pd.DataFrame(rows_summary), pd.DataFrame(rows_ranks), pd.DataFrame(rows_nemenyi))


def _write_master_files(all_fold, all_summary, all_sig, all_corr, all_epsq1, all_ranking_detail):
    """
    COLAB GUVENLIGI: orijinalde bu yalnizca run_full_pipeline sonunda BIR KEZ
    cagriliyordu. Burada her veri seti bittiginde YENIDEN cagirilir --
    Colab oturumu ortada koparsa bile o ana kadarki tum sonuclar diskte kalir.
    """
    master_summary = pd.concat(all_summary, ignore_index=True)

    pd.concat(all_fold, ignore_index=True).to_excel(
        os.path.join(MASTER_DIR, 'MASTER_FOLD_DETAIL.xlsx'), index=False)
    master_summary.to_excel(
        os.path.join(MASTER_DIR, 'MASTER_SUMMARY.xlsx'), index=False)
    pd.concat(all_sig, ignore_index=True).to_excel(
        os.path.join(MASTER_DIR, 'MASTER_SIGNIFICANCE.xlsx'), index=False)
    pd.concat(all_corr, ignore_index=True).to_excel(
        os.path.join(MASTER_DIR, 'MASTER_RANK_CORRELATION.xlsx'), index=False)
    pd.concat(all_epsq1, ignore_index=True).to_excel(
        os.path.join(MASTER_DIR, 'MASTER_EPSILON_Q1_LOG.xlsx'), index=False)
    pd.concat(all_ranking_detail, ignore_index=True).to_excel(
        os.path.join(MASTER_DIR, 'MASTER_V1_RANKING_DETAIL.xlsx'), index=False)

    print("  [checkpoint] MASTER_*.xlsx guncellendi "
          f"({len(all_summary)}/{len(DATASETS)} veri seti tamamlandi).")

    if len(all_summary) >= 3:
        try:
            fried_summary, fried_ranks, fried_nemenyi = run_friedman_nemenyi(master_summary)
            with pd.ExcelWriter(os.path.join(MASTER_DIR, 'MASTER_FRIEDMAN_NEMENYI.xlsx'), engine='openpyxl') as w:
                fried_summary.to_excel(w, sheet_name='Friedman_Summary', index=False)
                fried_ranks.to_excel(w, sheet_name='Average_Ranks', index=False)
                fried_nemenyi.to_excel(w, sheet_name='Nemenyi_Pairwise_p', index=False)
            print("  [checkpoint] MASTER_FRIEDMAN_NEMENYI.xlsx guncellendi.")
        except Exception as e:
            print(f"  [checkpoint] Friedman/Nemenyi henuz hesaplanamadi: {e!r}")


def run_full_pipeline():
    active_datasets = {
        k: v for k, v in DATASETS.items() if RUN_ONLY is None or k in RUN_ONLY
    }
    print(f"\nCalistirilacak veri setleri ({len(active_datasets)}): {list(active_datasets.keys())}")

    all_fold, all_summary, all_sig, all_corr, all_epsq1, all_ranking_detail = [], [], [], [], [], []

    t_start = time.time()
    for ds_name, (filename, target_hint) in active_datasets.items():
        ds_t0 = time.time()
        try:
            result = run_dataset(ds_name, filename, target_hint)
            all_fold.append(result['fold'])
            all_summary.append(result['summary'])
            all_sig.append(result['sig'])
            all_corr.append(result['corr'])
            all_epsq1.append(result['epsq1'])
            all_ranking_detail.append(result['ranking_detail'])

            # COLAB GUVENLIGI: her dataset sonrasi MASTER dosyalarini yaz/guncelle.
            if all_summary:
                _write_master_files(all_fold, all_summary, all_sig, all_corr, all_epsq1, all_ranking_detail)

            print(f"  >> [{ds_name}] toplam sure: {(time.time() - ds_t0) / 60:.1f} dk "
                  f"(kosumun tamami simdiye kadar: {(time.time() - t_start) / 60:.1f} dk)")
        except Exception as e:
            print(f"\n  HATA [{ds_name}]: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"PIPELINE TAMAMLANDI. Toplam sure: {(time.time() - t_start) / 60:.1f} dk")
    print(f"Sonuclar: {MASTER_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    prepare_all_new_datasets()
    run_full_pipeline()
