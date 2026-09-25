import numpy as np, pandas as pd
from src.prep import load_split, load_gt
from src.matching import select
s1,pool=load_split("devdata","train","cache"); gt,gtp=load_gt("devdata/train/train_ground_truth.tsv")
o=pd.read_parquet("cache/dev_oof.parquet")
sid={v:i for i,v in enumerate(s1.entity_id)}; pid={v:i for i,v in enumerate(pool.entity_id)}
T=pd.DataFrame({"s1":gtp.s1_id.map(sid),"pool":gtp.pool_id.map(pid)})
ntrue=T.groupby("s1").size().reindex(range(len(s1)),fill_value=0).values
sel=select(o.s1_idx.values,o.pool_idx.values,o.p.values,0.70)
P=sel[["s1","pool"]]
tp=P.merge(T).groupby("s1").size().reindex(range(len(s1)),fill_value=0).values
npred=P.groupby("s1").size().reindex(range(len(s1)),fill_value=0).values
incand=o[o.y==1].groupby("s1_idx").size().reindex(range(len(s1)),fill_value=0).values
prec=np.where(npred>0,tp/np.maximum(npred,1),0); rec=np.where(ntrue>0,tp/np.maximum(ntrue,1),0)
f=np.where(tp>0,1.25*prec*rec/(0.25*prec+rec+1e-12),0.); f=np.where((ntrue==0)&(npred==0),1.,f)
loss=1-f; N=len(s1)
print("macro F0.5 %.4f  total loss %.4f"%(f.mean(),loss.mean()))
cat=np.select([ntrue==0, npred==0, (npred-tp)>0, tp<ntrue],["singleton_FP","pred_empty_but_true","has_FP","only_FN"],"perfect")
df=pd.DataFrame({"cat":cat,"loss":loss,"fp":npred-tp,"fn":ntrue-tp,"fn_block":ntrue-incand})
g=df.groupby("cat").agg(n=("loss","size"),loss_sum=("loss","sum")); g["share_of_total_loss"]=g.loss_sum/g.loss_sum.sum(); g["pts"]=g.loss_sum/N
print(g.round(4))
# FN split: blocking vs model, weighted by F loss contribution attributable (count basis)
print("FN pairs: blocking-missed %d, model-missed %d ; FP pairs %d ; true pairs %d"%(df.fn_block.sum(), (df.fn-df.fn_block).sum(), df.fp.sum(), ntrue.sum()))
# counterfactuals
def F(tp,npred):
    prec=np.where(npred>0,tp/np.maximum(npred,1),0); rec=np.where(ntrue>0,tp/np.maximum(ntrue,1),0)
    f=np.where(tp>0,1.25*prec*rec/(0.25*prec+rec+1e-12),0.); return np.where((ntrue==0)&(npred==0),1.,f).mean()
print("if no FP at all: %.4f"%F(tp,tp))
print("if no model FN (all in-candidate trues found, FPs kept): %.4f"%F(incand,npred-tp+incand))
print("if perfect blocking+model recall, FPs kept: %.4f"%F(ntrue,npred-tp+ntrue))
