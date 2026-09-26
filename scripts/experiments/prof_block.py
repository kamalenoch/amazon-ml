import time, numpy as np, pandas as pd
import src.blocking as B
from src.prep import load_split
s1,pool=load_split("devdata","train","cache")
ix=B.KeyIndex(s1,pool,cap=1200,df_scale=15.2,verbose=False)
sr,sf,pos=ix.sr,ix.sf,ix.pos
m=sr<60000; r,f,p=sr[m],sf[m],pos[m]
T=time.time(); cnt=ix.counts[p]; tot=int(cnt.sum())
rep=np.repeat(np.arange(len(r)),cnt); offs=np.arange(tot)-np.repeat(np.cumsum(cnt)-cnt,cnt)+np.repeat(ix.starts[p],cnt)
pairs=(r[rep].astype(np.int64)<<24)|ix.pr[offs].astype(np.int64); w=ix.idf[p][rep]; fam=f[rep]; print("expand",tot,round(time.time()-T,2)); T=time.time()
up,inv=np.unique(pairs,return_inverse=True); print("np.unique",len(up),round(time.time()-T,2)); T=time.time()
codes,uniq=pd.factorize(pairs); print("pd.factorize",round(time.time()-T,2)); T=time.time()
sn=np.bincount(inv,weights=w*(fam==0),minlength=len(up)); print("bincount",round(time.time()-T,2)); T=time.time()
g=(up>>24).astype(np.int32); B._group_rank(g,sn.astype(np.float32)); print("group_rank(lexsort)",round(time.time()-T,2)); T=time.time()
# fast rank: up is sorted by g already -> sort within groups via combined key
v=sn.astype(np.float32); key=(g.astype(np.int64)<<32)|(np.int64(0x7FFFFFFF)-v.view(np.int32).astype(np.int64))
o=np.argsort(key,kind="stable"); print("combined-key argsort",round(time.time()-T,2))
