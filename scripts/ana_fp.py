import numpy as np, pandas as pd
from src.matching import select
kw=dict(sep="\t",dtype=str,keep_default_na=False,quoting=3)
r1=pd.read_csv("devdata/train/train_source1.tsv",**kw)
rp=pd.concat([pd.read_csv("devdata/train/train_source2.tsv",**kw),pd.read_csv("devdata/train/train_source3.tsv",**kw)],ignore_index=True)
gt=pd.read_csv("devdata/train/train_ground_truth.tsv",**kw)
owner={m:s for s,ms in zip(gt.source1_entity_id,gt.matched_entity_ids) for m in ms.split(",") if m}
o=pd.read_parquet("cache/dev_oof.parquet")
sel=select(o.s1_idx.values,o.pool_idx.values,o.p.values,0.70).merge(o[["s1_idx","pool_idx","y"]],left_on=["s1","pool"],right_on=["s1_idx","pool_idx"])
fp=sel[sel.y==0]
own=fp.pool.map(lambda j: owner.get(rp.entity_id.iloc[j]))
print("FP pairs",len(fp)," pool rec is a distractor (no owner):",own.isna().mean().round(3), " owner is another S1:",own.notna().mean().round(3))
print("FP p hist",np.histogram(fp.p,[.7,.8,.9,.95,.99,1.01])[0])
rng=np.random.default_rng(5)
for k in rng.choice(len(fp),12,replace=False):
    r=fp.iloc[k]; a=r1.iloc[int(r.s1)]; b=rp.iloc[int(r.pool)]; ow=own.iloc[k]
    print(f"\np={r.p:.2f} S1: {a.business_name} | {a.business_address}\n   FP: {b.business_name} | {b.business_address}")
    if ow is not None:
        c=r1[r1.entity_id==ow]
        if len(c): print(f"   TRUE OWNER: {c.business_name.iloc[0]} | {c.business_address.iloc[0]}")
        else: print("   TRUE OWNER outside dev subset")
    else: print("   (distractor)")
