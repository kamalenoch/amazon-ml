import time, sys, numpy as np, pandas as pd
import src.blocking as B
from src.prep import load_split, load_gt
s1,pool=load_split("devdata","train","cache"); gt,gtp=load_gt("devdata/train/train_ground_truth.tsv")
key=pd.MultiIndex.from_arrays([gtp.s1_id,gtp.pool_id])
for cs,cc in [tuple(map(int,sys.argv[1:3]))]:
    t=time.time(); ix=B.KeyIndex(s1,pool,cap=cs,cap_composite=cc,df_scale=15.2,verbose=False); exp=ix.counts[ix.pos].sum()
    c=B.generate_candidates(s1,pool,k_total=150,k_name=40,k_addr=40,k_cross=40,index=ix,verbose=False)
    y=pd.MultiIndex.from_arrays([s1.entity_id.values[c.s1_idx],pool.entity_id.values[c.pool_idx]]).isin(key)
    print(f"cap single={cs} composite={cc}: expansion/S1 {exp/len(s1):.0f} (full ~{exp/len(s1)*15.2:.0f}) wide pairs/S1 {len(c)/len(s1):.1f} wide recall {y.sum()/len(gtp):.4f} ({time.time()-t:.0f}s)",flush=True)
    del ix,c
