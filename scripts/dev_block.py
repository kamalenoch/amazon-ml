import time, sys, numpy as np, pandas as pd
from src.prep import load_split, load_gt
from src.blocking import generate_candidates
t=time.time()
s1,pool=load_split("devdata","train","cache")
print("loaded",len(s1),len(pool),time.time()-t)
gt,gtp=load_gt("devdata/train/train_ground_truth.tsv")
kw=eval("dict(%s)"%(sys.argv[1] if len(sys.argv)>1 else ""))
c=generate_candidates(s1,pool,**kw)
c["s1_id"]=s1.entity_id.values[c.s1_idx]; c["pool_id"]=pool.entity_id.values[c.pool_idx]
m=gtp.merge(c[["s1_id","pool_id"]].assign(hit=1),how="left")
print(kw,"pairs",len(c),"per s1 %.1f"%(len(c)/len(s1)),"RECALL %.4f"%m.hit.notna().mean(), "time",time.time()-t)
c.to_parquet("cache/dev_cands.parquet")
