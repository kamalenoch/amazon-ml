import time, numpy as np, pandas as pd
from src.siblings import sibling_features
from src.matching import train_lgb, feat_cols, select, f05_macro
from src.prep import load_gt
t=time.time()
XM2=pd.read_parquet("cache/dev_XM2.parquet"); p1=np.load("cache/oof_noise.npy")
pool=pd.read_parquet("cache/train_norm_pool.parquet"); s1=pd.read_parquet("cache/train_norm_s1.parquet")
S=sibling_features(XM2[["s1_idx","pool_idx"]],p1,pool); print("sib feats",S.shape,time.time()-t)
X3=pd.concat([XM2,S],axis=1)
gt,gtp=load_gt("devdata/train/train_ground_truth.tsv")
ids=s1.entity_id.values; pids=pool.entity_id.values
Ms=set(ids[np.unique(X3.s1_idx)]); true=gtp[gtp.s1_id.isin(Ms)].rename(columns={"s1_id":"s1","pool_id":"pool"})
fold=(X3.s1_idx.values//2)%4
F=feat_cols(X3); oof=np.zeros(len(X3),np.float32)
for k in range(4):
    tr,va=fold!=k,fold==k
    m=train_lgb(X3.loc[tr,F],X3.y[tr],rounds=400); oof[va]=m.predict(X3.loc[va,F])
np.save("cache/oof_sib.npy",oof)
for thr in [0.4,0.5,0.6,0.7,0.8,0.9]:
    sel=select(X3.s1_idx.values,X3.pool_idx.values,oof,thr)
    pred=pd.DataFrame({"s1":ids[sel.s1],"pool":pids[sel.pool]})
    print("sib",thr,f05_macro(pred,true,list(Ms)),flush=True)
imp=pd.Series(m.feature_importance("gain"),F).sort_values(ascending=False); print(imp.head(15).round(0).to_dict())
print(time.time()-t)
