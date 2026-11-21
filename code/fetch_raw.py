"""5 küçük veri setinin HAM (winsorize/korelasyon filtresi uygulanmamış) hâllerini indirir → data_raw/*.csv
ve READY_FOR_RS sürümleriyle satır/sütun eşleşmesini raporlar."""
import os, io, zipfile, urllib.request, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_openml
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,'..','data_raw'); READY=os.path.join(HERE,'..','..','data')
os.makedirs(OUT,exist_ok=True)
def save(name, X, y):
    df=X.copy(); df['target']=np.asarray(y).astype(int); df.to_csv(os.path.join(OUT,f'{name}_raw.csv'),index=False); return df
bc=load_breast_cancer(as_frame=True); save('kanser', bc.data, bc.target)
db=load_diabetes(as_frame=True); r=pd.read_excel(os.path.join(READY,'diabetdata3_READY_FOR_RS.xlsx'))
# Risk_Class eşiğini bul: READY'deki 1'lerin ham hedef aralığı
t=db.target.values; lab=r.Risk_Class.values
thr=(t[lab==1].min(), t[lab==0].max()); print('diabetes Risk_Class: min(target|1)=',thr[0],' max(target|0)=',thr[1])
save('diabet', db.data, (t>=thr[0]).astype(int))
def uci(url):
    return zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url,timeout=60).read()))
z=uci('https://archive.ics.uci.edu/static/public/151/connectionist+bench+sonar+mines+vs+rocks.zip'); print('sonar zip:',z.namelist())
so=pd.read_csv(io.BytesIO(z.read([n for n in z.namelist() if n.endswith('sonar.all-data')][0])),header=None)
save('sonar', so.iloc[:,:60].set_axis([f'feature_{i}' for i in range(60)],axis=1), (so[60]=='M').astype(int))
z=uci('https://archive.ics.uci.edu/static/public/174/parkinsons.zip'); print('pk zip:',z.namelist())
pk=pd.read_csv(io.BytesIO(z.read([n for n in z.namelist() if n.endswith('parkinsons.data')][0])))
save('parkinson', pk.drop(columns=['name','status']), pk['status'])
url='https://archive.ics.uci.edu/static/public/429/cryotherapy+dataset.zip'
z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url,timeout=60).read())); print('cryo zip:',z.namelist())
nm=[n for n in z.namelist() if n.lower().endswith(('.xlsx','.xls'))][0]
cr=pd.read_excel(io.BytesIO(z.read(nm))); print(cr.columns.tolist())
save('cyron', cr.drop(columns=['Result_of_Treatment']), cr['Result_of_Treatment'])
# --- READY ile karşılaştırma
spec={'kanser':'kansersondata3','diabet':'diabetdata3','sonar':'sonar','parkinson':'parkinsondata3','cyron':'cyrondata3'}
for k,f in spec.items():
    raw=pd.read_csv(os.path.join(OUT,f'{k}_raw.csv')); rd=pd.read_excel(os.path.join(READY,f'{f}_READY_FOR_RS.xlsx'))
    print(f'{k:10s} raw {raw.shape} pos={raw.target.sum()} | ready {rd.shape} pos-col last={rd.iloc[:,-1].value_counts().to_dict()}')
