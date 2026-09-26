import time, numpy as np, pandas as pd
from src.prep import load_split, load_gt
from src.matching import train_lgb, feat_cols, select, f05_macro
t=time.time()
s1,pool=load_split("devdata","train","cache"); gt,gtp=load_gt("devdata/train/train_ground_truth.tsv")
X=pd.read_parquet("cache/dev_X.parquet")
fold=(X.s1_idx.values.astype(np.int64)*2654435761 % 2**32) % 4
F=feat_cols(X); oof=np.zeros(len(X),np.float32)
for k in range(4):
    tr,va=fold!=k,fold==k
    m=train_lgb(X.loc[tr,F],X.y[tr],rounds=400)
    oof[va]=m.predict(X.loc[va,F]); print("fold",k,time.time()-t,flush=True)
X["p"]=oof; X[["s1_idx","pool_idx","y","p"]].to_parquet("cache/dev_oof.parquet")
imp=pd.Series(m.feature_importance("gain"),F).sort_values(ascending=False); print(imp.head(20).round(0).to_dict())
true=pd.DataFrame({"s1":gtp.s1_id,"pool":gtp.pool_id})
for oto in [False,True]:
  for thr in [0.3,0.4,0.5,0.6,0.7]:
    sel=select(X.s1_idx.values,X.pool_idx.values,oof,thr,oto)
    pred=pd.DataFrame({"s1":s1.entity_id.values[sel.s1],"pool":pool.entity_id.values[sel.pool]})
    print(oto,thr,f05_macro(pred,true,s1.entity_id))
