import time, numpy as np, pandas as pd
from src.noise import raw_tokens, raw_numbers, learn_logodds, token_noise_feats, number_feats, TOKEN_COLS, NUM_COLS
from src.matching import train_lgb, feat_cols, select, f05_macro
from src.prep import load_gt
t=time.time()
kw=dict(sep="\t",dtype=str,keep_default_na=False,quoting=3)
r1=pd.read_csv("devdata/train/train_source1.tsv",**kw)
rp=pd.concat([pd.read_csv("devdata/train/train_source2.tsv",**kw),pd.read_csv("devdata/train/train_source3.tsv",**kw)],ignore_index=True)
N1=[raw_tokens(s) for s in r1.business_name]; NP=[raw_tokens(s) for s in rp.business_name]
A1=[raw_tokens(s) for s in r1.business_address]; AP=[raw_tokens(s) for s in rp.business_address]
U1=[raw_numbers(s) for s in r1.business_address]; UP=[raw_numbers(s) for s in rp.business_address]
print("tok",time.time()-t)
X=pd.read_parquet("cache/dev_X.parquet")
h=(X.s1_idx.values.astype(np.int64)*2654435761 % 2**32)
D=(h%2)==0; M=~D
ii,jj,y=X.s1_idx.values,X.pool_idx.values,X.y.values
dD=np.flatnonzero(D)
nadd,ndrop=learn_logodds([N1[i] for i in ii[dD]],[NP[j] for j in jj[dD]],y[dD])
aadd,adrop=learn_logodds([A1[i] for i in ii[dD]],[AP[j] for j in jj[dD]],y[dD])
print("dicts",len(nadd),len(ndrop),len(aadd),len(adrop),time.time()-t)
mM=np.flatnonzero(M)
F1=np.array([token_noise_feats(N1[i],NP[j],nadd,ndrop) for i,j in zip(ii[mM],jj[mM])],np.float32)
F2=np.array([token_noise_feats(A1[i],AP[j],aadd,adrop) for i,j in zip(ii[mM],jj[mM])],np.float32)
F3=np.array([number_feats(U1[i],UP[j]) for i,j in zip(ii[mM],jj[mM])],np.float32)
XM=X.iloc[mM].reset_index(drop=True)
new=pd.concat([pd.DataFrame(F1,columns=["nn_"+c for c in TOKEN_COLS]),pd.DataFrame(F2,columns=["an_"+c for c in TOKEN_COLS]),pd.DataFrame(F3,columns=NUM_COLS)],axis=1)
XM2=pd.concat([XM,new],axis=1); XM2.to_parquet("cache/dev_XM2.parquet")
print("feats",time.time()-t)
gt,gtp=load_gt("devdata/train/train_ground_truth.tsv")
ids=r1.entity_id.values; pids=rp.entity_id.values
Ms=set(ids[np.unique(XM.s1_idx)])
true=gtp[gtp.s1_id.isin(Ms)].rename(columns={"s1_id":"s1","pool_id":"pool"})
fold=(XM.s1_idx.values.astype(np.int64)*40503 % 2**31)%4
for name,XX in [("base",XM),("noise",XM2)]:
    F=feat_cols(XX); oof=np.zeros(len(XX),np.float32)
    for k in range(4):
        tr,va=fold!=k,fold==k
        m=train_lgb(XX.loc[tr,F],XX.y[tr],rounds=400); oof[va]=m.predict(XX.loc[va,F])
    np.save(f"cache/oof_{name}.npy",oof)
    for thr in [0.5,0.6,0.7,0.8]:
        sel=select(XX.s1_idx.values,XX.pool_idx.values,oof,thr)
        pred=pd.DataFrame({"s1":ids[sel.s1],"pool":pids[sel.pool]})
        print(name,thr,f05_macro(pred,true,list(Ms)),flush=True)
    if name=="noise":
        imp=pd.Series(m.feature_importance("gain"),F).sort_values(ascending=False); print(imp.head(15).round(0).to_dict())
print(time.time()-t)
