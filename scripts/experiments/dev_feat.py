import time, numpy as np, pandas as pd
from src.prep import load_split, load_gt
from src.features import build_idf, all_features
t=time.time()
s1,pool=load_split("devdata","train","cache")
c=pd.read_parquet("cache/dev_cands.parquet")
idf_n=build_idf([s1.n_clean,pool.n_clean]); idf_a=build_idf([s1.a_clean,pool.a_clean]); print("idf",time.time()-t)
X=all_features(c,s1,pool,idf_n,idf_a); print("feats",X.shape,time.time()-t)
gt,gtp=load_gt("devdata/train/train_ground_truth.tsv")
key=pd.Series(1,index=pd.MultiIndex.from_arrays([gtp.s1_id,gtp.pool_id]))
y=pd.MultiIndex.from_arrays([s1.entity_id.values[c.s1_idx],pool.entity_id.values[c.pool_idx]]).isin(key.index)
X["y"]=y.astype(np.int8); X["s1_idx"]=c.s1_idx.values; X["pool_idx"]=c.pool_idx.values
X.to_parquet("cache/dev_X.parquet"); print(X.y.mean(), time.time()-t)
