import re, numpy as np, pandas as pd
from collections import Counter
kw=dict(sep="\t",dtype=str,keep_default_na=False,quoting=3)
r1=pd.read_csv("devdata/train/train_source1.tsv",**kw)
rp=pd.concat([pd.read_csv("devdata/train/train_source2.tsv",**kw),pd.read_csv("devdata/train/train_source3.tsv",**kw)],ignore_index=True)
o=pd.read_parquet("cache/dev_oof.parquet")
R=re.compile(r"[0-9#]+")
def nums(s): return [t for t in R.findall(s) if any(c.isdigit() for c in t) or len(t)>1]
def rel(a,b):
    # relation of best-aligned number pair between S1 token list a and pool list b
    if not a or not b: return "missing"
    best="unrelated"; order=["equal","zero_pad","masked","deletion","insertion","transpose","substitution","unrelated"]
    for x in a:
        for y in b:
            if y==x: r="equal"
            elif y.lstrip("0")==x.lstrip("0"): r="zero_pad"
            elif "#" in y and len(y)==len(x) and all(c==d or c=="#" for d,c in zip(x,y)): r="masked"
            elif "#" in y and y.replace("#","") and y.replace("#","").lstrip("0") in x: r="masked"
            elif len(y)==len(x)-1 and any(x[:i]+x[i+1:]==y for i in range(len(x))): r="deletion"
            elif len(y)==len(x)+1 and any(y[:i]+y[i+1:]==x for i in range(len(y))): r="insertion"
            elif len(y)==len(x) and sum(c!=d for c,d in zip(x,y))==2 and sorted(x)==sorted(y): r="transpose"
            elif len(y)==len(x) and sum(c!=d for c,d in zip(x,y))==1: r="substitution"
            else: r="unrelated"
            if order.index(r)<order.index(best): best=r
    return best
def dist(df,n=40000):
    df=df.sample(min(n,len(df)),random_state=0)
    return Counter(rel(nums(r1.business_address.iloc[i]),nums(rp.business_address.iloc[j])) for i,j in zip(df.s1_idx,df.pool_idx))
pos=o[o.y==1]; hard=o[(o.y==0)&(o.p>0.3)]; fp=o[(o.y==0)&(o.p>0.7)]
for name,df in [("true pairs",pos),("hard neg p>0.3",hard),("FP p>0.7",fp)]:
    c=dist(df); t=sum(c.values()); print(f"{name:16s}", {k:round(v/t,3) for k,v in c.most_common()})
