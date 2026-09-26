import time, numpy as np, pandas as pd
import src.blocking as B
from src.prep import load_split, load_gt
from src.prerank import prerank_features
import lightgbm as lgb
s1,pool=load_split("devdata","train","cache"); gt,gtp=load_gt("devdata/train/train_ground_truth.tsv")
key=pd.MultiIndex.from_arrays([gtp.s1_id,gtp.pool_id])
for cap in [1200,2400]:
    t=time.time(); ix=B.KeyIndex(s1,pool,cap=cap,df_scale=15.2,verbose=False); exp=ix.counts[ix.pos].sum()
    c=B.generate_candidates(s1,pool,k_total=150,k_name=40,k_addr=40,k_cross=40,index=ix,verbose=False)
    c["y"]=pd.MultiIndex.from_arrays([s1.entity_id.values[c.s1_idx],pool.entity_id.values[c.pool_idx]]).isin(key).astype(np.int8)
    X=prerank_features(c); h=(c.s1_idx.values//2)%2
    m=lgb.train(dict(objective="binary",learning_rate=0.1,num_leaves=63,min_data_in_leaf=200,verbose=-1),lgb.Dataset(X[h==0],c.y[h==0]),300)
    cc=c[h==1].copy(); cc["s"]=m.predict(X[h==1]); nt1=gtp.s1_id.isin(set(s1.entity_id.values[np.unique(cc.s1_idx)])).sum()
    r=B._group_rank(cc.s1_idx.values,cc.s.values)
    print(f"cap={cap} expansion/S1 {exp/len(s1):.0f} (full ~{exp/len(s1)*15.2:.0f}) wide pairs/S1 {len(c)/len(s1):.1f} wide recall {c.y.sum()/len(gtp):.4f} | prerank recall@20 {cc.y[r<20].sum()/nt1:.4f} @30 {cc.y[r<30].sum()/nt1:.4f} @40 {cc.y[r<40].sum()/nt1:.4f} ({time.time()-t:.0f}s)",flush=True)
