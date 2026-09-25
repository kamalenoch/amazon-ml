import numpy as np, pandas as pd
from src.prep import load_split, load_gt
from src.blocking import KeyIndex, generate_candidates, record_keys
s1,pool=load_split("devdata","train","cache"); gt,gtp=load_gt("devdata/train/train_ground_truth.tsv")
kw=dict(sep="\t",dtype=str,keep_default_na=False,quoting=3)
r1=pd.read_csv("devdata/train/train_source1.tsv",**kw).set_index("entity_id")
rp=pd.concat([pd.read_csv("devdata/train/train_source2.tsv",**kw),pd.read_csv("devdata/train/train_source3.tsv",**kw)]).set_index("entity_id")
ix=KeyIndex(s1,pool,cap=600,df_scale=15.2,verbose=False)
c=generate_candidates(s1,pool,k_total=150,k_name=40,k_addr=40,k_cross=40,index=ix,verbose=False)
have=set(zip(s1.entity_id.values[c.s1_idx],pool.entity_id.values[c.pool_idx]))
miss=gtp[[ (a,b) not in have for a,b in zip(gtp.s1_id,gtp.pool_id)]]
print("missed",len(miss))
S=s1.set_index("entity_id"); P=pool.set_index("entity_id")
cats=[]
for a,b in zip(miss.s1_id,miss.pool_id):
    x=S.loc[a]; y=P.loc[b]; k1=record_keys(x.country.lower(),x.n_clean,x.a_clean,x.nums); k2=record_keys(y.country.lower(),y.n_clean,y.a_clean,y.nums)
    cats.append((len(k1[0]&k2[0]),len(k1[1]&k2[1]),len(k1[2]&k2[2]), y.a_clean=="", y.n_clean.count(" ")==0))
d=pd.DataFrame(cats,columns=["name","addr","cross","pool_addr_empty","pool_single_tok"])
print("no shared key at all: %.3f | pool addr empty: %.3f | single-token pool name: %.3f"%(((d.name+d.addr+d.cross)==0).mean(),d.pool_addr_empty.mean(),d.pool_single_tok.mean()))
print("shared keys exist but all capped (df too high):",(((d.name+d.addr+d.cross)>0)).mean().round(3))
for a,b in list(zip(miss.s1_id,miss.pool_id))[:30]:
    print(f"{r1.loc[a].business_name[:38]:38s} | {r1.loc[a].business_address[:50]:50s}\n{rp.loc[b].business_name[:38]:38s} | {rp.loc[b].business_address[:50]}\n")
