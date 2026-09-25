import numpy as np, pandas as pd
from src.prep import load_split, load_gt
from src.matching import select
s1,pool=load_split("devdata","train","cache"); gt,gtp=load_gt("devdata/train/train_ground_truth.tsv")
X=pd.read_parquet("cache/dev_XM2.parquet",columns=["s1_idx","pool_idx","y"]); X["p"]=np.load("cache/oof_noise.npy")
S=np.unique(X.s1_idx); sid={v:i for i,v in enumerate(s1.entity_id)}; pid={v:i for i,v in enumerate(pool.entity_id)}
T=pd.DataFrame({"s1":gtp.s1_id.map(sid),"pool":gtp.pool_id.map(pid)}); T=T[T.s1.isin(S)]
ntrue=T.groupby("s1").size().reindex(S,fill_value=0).values
incand=X[X.y==1].groupby("s1_idx").size().reindex(S,fill_value=0).values
def F(tp,npred):
    prec=np.where(npred>0,tp/np.maximum(npred,1),0); rec=np.where(ntrue>0,tp/np.maximum(ntrue,1),0)
    f=np.where(tp>0,1.25*prec*rec/(0.25*prec+rec+1e-12),0.); return np.where((ntrue==0)&(npred==0),1.,f)
sel=select(X.s1_idx.values,X.pool_idx.values,X.p.values,0.8)
P=sel[["s1","pool"]]
tp=P.merge(T).groupby("s1").size().reindex(S,fill_value=0).values
npred=P.groupby("s1").size().reindex(S,fill_value=0).values
f=F(tp,npred); print("F %.4f"%f.mean())
print("if blocking perfect (model recall rate on in-cand trues applied): ~", "blocking-missed pairs %d / %d true (%.2f%%)"%((ntrue-incand).sum(),ntrue.sum(),100*(ntrue-incand).sum()/ntrue.sum()))
print("model-missed in-cand pairs %d, FP pairs %d"%((incand-tp).sum(),(npred-tp).sum()))
print("upper bound, no FP: %.4f | no model-FN: %.4f | no blocking-FN (all found): %.4f"%(F(tp,tp).mean(),F(incand,npred-tp+incand).mean(),F(tp+(ntrue-incand),npred+(ntrue-incand)).mean()))
