import time, numpy as np, pandas as pd
from src.prep import load_split, load_gt
from src.matching import train_lgb, feat_cols, select, f05_macro
from src.features import context_features
t=time.time()
s1,pool=load_split("devdata","train","cache"); gt,gtp=load_gt("devdata/train/train_ground_truth.tsv")
X=pd.read_parquet("cache/dev_X.parquet"); o=pd.read_parquet("cache/dev_oof.parquet"); X["p1"]=o.p.values
C=context_features(X[["s1_idx","pool_idx","p1"]],"p1","p1")
X=pd.concat([X,C],axis=1)
g=X.groupby("s1_idx").p1
X["p1_sum_s1"]=g.transform("sum").astype(np.float32); X["p1_n05_s1"]=g.transform(lambda s:(s>0.5).sum()).astype(np.float32)
fold=(X.s1_idx.values.astype(np.int64)*2654435761 % 2**32) % 4
F=feat_cols(X); oof=np.zeros(len(X),np.float32)
for k in range(4):
    tr,va=fold!=k,fold==k
    m=train_lgb(X.loc[tr,F],X.y[tr],rounds=300); oof[va]=m.predict(X.loc[va,F])
X[["s1_idx","pool_idx","y"]].assign(p=oof).to_parquet("cache/dev_oof2.parquet")
true=pd.DataFrame({"s1":gtp.s1_id,"pool":gtp.pool_id})
for thr in [0.4,0.5,0.6,0.7,0.8]:
    sel=select(X.s1_idx.values,X.pool_idx.values,oof,thr,True)
    pred=pd.DataFrame({"s1":s1.entity_id.values[sel.s1],"pool":pool.entity_id.values[sel.pool]})
    print(thr,f05_macro(pred,true,s1.entity_id),flush=True)
print(time.time()-t)
