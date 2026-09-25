import numpy as np, pandas as pd
from src.prep import load_split, load_gt
from src.matching import select, f05_macro
from src.select import expected_f_select
s1,pool=load_split("devdata","train","cache"); gt,gtp=load_gt("devdata/train/train_ground_truth.tsv")
X=pd.read_parquet("cache/dev_XM2.parquet",columns=["s1_idx","pool_idx"]); p=np.load("cache/oof_noise.npy")
ids=s1.entity_id.values; pids=pool.entity_id.values
Ms=set(ids[np.unique(X.s1_idx)]); true=gtp[gtp.s1_id.isin(Ms)].rename(columns={"s1_id":"s1","pool_id":"pool"})
def ev(sel): return f05_macro(pd.DataFrame({"s1":ids[sel.s1.values],"pool":pids[sel.pool.values]}),true,list(Ms))[0]
print("threshold 0.8:", round(ev(select(X.s1_idx.values,X.pool_idx.values,p,0.8)),4))
for mp in [0.0,0.1,0.2]:
    for pw in [1.0,1.5,2.0]:
        print(f"expF miss_prior={mp} power={pw}:", round(ev(expected_f_select(X.s1_idx.values,X.pool_idx.values,p,miss_prior=mp,power=pw)),4))
