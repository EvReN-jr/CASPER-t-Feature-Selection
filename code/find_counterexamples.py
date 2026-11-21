"""Teorik önermeler için küçük karşı örnekleri kaba kuvvetle arar."""
import numpy as np
from casper_core import pairwise, gamma_op, gamma_near, gamma_tol
rng=np.random.RandomState(0)
found={}
for it in range(200000):
    n=rng.randint(3,6); p=rng.randint(1,3)
    X=rng.randint(0,4,size=(n,p)).astype(float); y=rng.randint(0,2,size=n)
    if len(set(y))<2: continue
    D=pairwise(X)
    # (a) gamma_op non-monotone in eps: g(e1) < g(e2) for e1<e2 while g(e2)<1 ... any increase
    if 'eps' not in found:
        gs=[gamma_op(D,y,e) for e in range(0,7)]
        for e in range(6):
            if gs[e+1]>gs[e]+1e-9 and gs[e]>0:
                found['eps']=(X.tolist(),y.tolist(),e,gs); break
    # (b) attribute non-monotone for gamma_op: adding a column decreases gamma at same eps
    if p==1 and 'attr' not in found:
        c=rng.randint(0,4,size=(n,1)).astype(float); X2=np.hstack([X,c]); D2=pairwise(X2)
        for e in range(0,4):
            if gamma_op(D2,y,e) < gamma_op(D,y,e)-1e-9:
                found['attr']=(X.tolist(),X2.tolist(),y.tolist(),e,gamma_op(D,y,e),gamma_op(D2,y,e)); break
    if p==1 and 'attr_near' not in found:
        c=rng.randint(0,4,size=(n,1)).astype(float); X2=np.hstack([X,c]); D2=pairwise(X2)
        for e in range(0,4):
            if gamma_near(D2,y,e) < gamma_near(D,y,e)-1e-9:
                found['attr_near']=(X.tolist(),X2.tolist(),y.tolist(),e,gamma_near(D,y,e),gamma_near(D2,y,e)); break
    if p==1 and 'tol_violation' not in found:
        c=rng.randint(0,4,size=(n,1)).astype(float); X2=np.hstack([X,c]); D2=pairwise(X2)
        for e in range(0,4):
            if gamma_tol(D2,y,e) < gamma_tol(D,y,e)-1e-9: found['tol_violation']=1
    if len([k for k in found if k!='tol_violation'])==3 and it>50000: break
for k,v in found.items(): print(k, v)
print('tol monotonicity violated?', 'tol_violation' in found)
