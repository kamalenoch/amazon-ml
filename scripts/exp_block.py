import time, numpy as np, pandas as pd, lightgbm as lgb
from src.prep import load_split, load_gt
from src.blocking import KeyIndex, generate_candidates, _group_rank
from src.prerank import prerank_features, PR_COLS
t=time.time()
s1,pool=load_split("devdata","train","cache"); gt,gtp=load_gt("devdata/train/train_ground_truth.tsv")
ix=KeyIndex(s1,pool,cap=600,df_scale=15.2)
c=generate_candidates(s1,pool,k_total=150,k_name=40,k_addr=40,k_cross=40,index=ix,verbose=False)
key=pd.MultiIndex.from_arrays([gtp.s1_id,gtp.pool_id])
c["y"]=pd.MultiIndex.from_arrays([s1.entity_id.values[c.s1_idx],pool.entity_id.values[c.pool_idx]]).isin(key).astype(np.int8)
ntrue=len(gtp); print("wide: pairs/S1 %.1f recall %.4f (%.0fs)"%(len(c)/len(s1),c.y.sum()/ntrue,time.time()-t))
X=prerank_features(c); h=(c.s1_idx.values//2)%2
m=lgb.train(dict(objective="binary",learning_rate=0.1,num_leaves=63,min_data_in_leaf=200,verbose=-1),lgb.Dataset(X[h==0],c.y[h==0]),300)
s=m.predict(X[h==1]); cc=c[h==1].copy(); cc["s"]=s
nt1=gtp.s1_id.isin(set(s1.entity_id.values[np.unique(cc.s1_idx)])).sum()
tot=(cc.bs_name+cc.bs_addr+cc.bs_cross).values
r_sum=_group_rank(cc.s1_idx.values,tot); r_pr=_group_rank(cc.s1_idx.values,s)
for K in [10,15,20,25,30,40,60]:
    print(f"K={K:3d}  recall sum-rank {cc.y[r_sum<K].sum()/nt1:.4f}   prerank {cc.y[r_pr<K].sum()/nt1:.4f}")
print("ceiling(wide) %.4f"%(cc.y.sum()/nt1))
imp=pd.Series(m.feature_importance("gain"),PR_COLS).sort_values(ascending=False); print(imp.round(0).to_dict())
m.save_model("cache/prerank_dev.txt")
