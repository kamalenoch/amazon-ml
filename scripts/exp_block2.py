import sys, time, numpy as np, pandas as pd
import src.blocking as B
from src.prep import load_split, load_gt
s1,pool=load_split("devdata","train","cache"); gt,gtp=load_gt("devdata/train/train_ground_truth.tsv")
key=pd.MultiIndex.from_arrays([gtp.s1_id,gtp.pool_id])
for extra in [False, True]:
    B.EXTRA_KEYS=extra; t=time.time()
    ix=B.KeyIndex(s1,pool,cap=600,df_scale=15.2,verbose=False)
    exp=ix.counts[ix.pos].sum()
    for kt,kn,ka,kx in [(30,10,10,0),(150,40,40,40)]:
        c=B.generate_candidates(s1,pool,k_total=kt,k_name=kn,k_addr=ka,k_cross=kx,index=ix,verbose=False)
        y=pd.MultiIndex.from_arrays([s1.entity_id.values[c.s1_idx],pool.entity_id.values[c.pool_idx]]).isin(key)
        print(f"extra={extra} K=({kt},{kn},{ka},{kx}) expansion/S1 {exp/len(s1):.0f} (x15.2 full ~{exp/len(s1)*15.2:.0f})  pairs/S1 {len(c)/len(s1):.1f}  recall {y.sum()/len(gtp):.4f}  ({time.time()-t:.0f}s)",flush=True)
