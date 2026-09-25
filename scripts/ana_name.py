import re, numpy as np, pandas as pd
from collections import Counter
from unidecode import unidecode
kw=dict(sep="\t",dtype=str,keep_default_na=False,quoting=3)
r1=pd.read_csv("devdata/train/train_source1.tsv",**kw)
rp=pd.concat([pd.read_csv("devdata/train/train_source2.tsv",**kw),pd.read_csv("devdata/train/train_source3.tsv",**kw)],ignore_index=True)
o=pd.read_parquet("cache/dev_oof.parquet")
T=lambda s: set(re.findall(r"[a-z0-9]+", unidecode(s).lower()))
def diffs(df,n=60000):
    df=df.sample(min(n,len(df)),random_state=0); add=Counter(); drop=Counter(); k=0
    for i,j in zip(df.s1_idx,df.pool_idx):
        a=T(r1.business_name.iloc[i]); b=T(rp.business_name.iloc[j])
        if not b.isascii() if isinstance(b,str) else False: continue
        add.update(b-a); drop.update(a-b); k+=1
    return add,drop,k
pa,pd_,npos=diffs(o[o.y==1]); na,nd,nneg=diffs(o[(o.y==0)&(o.p>0.3)])
def lo(c1,n1,c2,n2,top=25):
    keys=set(k for k,_ in c1.most_common(300))|set(k for k,_ in c2.most_common(300))
    r=sorted(((np.log((c1[k]+1)/n1)-np.log((c2[k]+1)/n2),k,c1[k],c2[k]) for k in keys))
    return r[:top], r[-top:]
neg,pos=lo(pa,npos,na,nneg)
print("ADDED tokens — most NEGATIVE-indicative (logodds,tok,pos,neg):",[(round(a,1),k,p,q) for a,k,p,q in neg])
print("ADDED tokens — most POSITIVE-indicative:",[(round(a,1),k,p,q) for a,k,p,q in pos])
neg,pos=lo(pd_,npos,nd,nneg)
print("DROPPED tokens — most NEGATIVE-indicative:",[(round(a,1),k,p,q) for a,k,p,q in neg])
print("DROPPED tokens — most POSITIVE-indicative:",[(round(a,1),k,p,q) for a,k,p,q in pos])
