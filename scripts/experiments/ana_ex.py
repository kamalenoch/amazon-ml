import numpy as np, pandas as pd
kw=dict(sep="\t",dtype=str,keep_default_na=False,quoting=3)
r1=pd.read_csv("devdata/train/train_source1.tsv",**kw)
rp=pd.concat([pd.read_csv("devdata/train/train_source2.tsv",**kw),pd.read_csv("devdata/train/train_source3.tsv",**kw)],ignore_index=True)
o=pd.read_parquet("cache/dev_oof.parquet")
best=o.groupby("pool_idx").p.transform("max")
fn=o[(o.y==1)&(o.p<0.7)]
print("model-missed p distribution:",np.histogram(fn.p,[0,.05,.2,.4,.6,.7])[0], " of which pool rec's best S1 is other:",(fn.p<best[fn.index]).mean().round(3))
rng=np.random.default_rng(3)
for k in rng.choice(fn.index,14,replace=False):
    i,j,p=o.s1_idx[k],o.pool_idx[k],o.p[k]
    a=r1.iloc[i]; b=rp.iloc[j]
    # siblings: other true matches of this s1 with high p
    sib=o[(o.s1_idx==i)&(o.y==1)&(o.pool_idx!=j)].sort_values("p",ascending=False).head(2)
    comp=o[(o.pool_idx==j)&(o.s1_idx!=i)].sort_values("p",ascending=False).head(1)
    print(f"\np={p:.2f} S1: {a.business_name} | {a.business_address}")
    print(f"   MISS: {b.business_name} | {b.business_address}")
    for _,s in sib.iterrows(): print(f"   sib p={s.p:.2f}: {rp.iloc[int(s.pool_idx)].business_name} | {rp.iloc[int(s.pool_idx)].business_address}")
    for _,s in comp.iterrows(): print(f"   rival S1 p={s.p:.2f}: {r1.iloc[int(s.s1_idx)].business_name} | {r1.iloc[int(s.s1_idx)].business_address}")
