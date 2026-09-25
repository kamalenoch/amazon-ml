# Paste into a Kaggle cell. Inputs: challenge dataset + the v1 notebook output (for test_scores.parquet).
import glob, os, numpy as np, pandas as pd
sc_path = glob.glob("/kaggle/input/**/test_scores.parquet", recursive=True)
sc_path = sc_path[0] if sc_path else "/kaggle/working/work/test_scores.parquet"
D = os.path.dirname(glob.glob("/kaggle/input/**/test_source1.tsv", recursive=True)[0])
kw = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3)
s1 = pd.read_csv(f"{D}/test_source1.tsv", **kw)
pool = pd.concat([pd.read_csv(f"{D}/test_source2.tsv", **kw), pd.read_csv(f"{D}/test_source3.tsv", **kw)], ignore_index=True)
sc = pd.read_parquet(sc_path)
sc["country"] = s1.country.values[sc.s1_idx.values]

best = sc.groupby("pool_idx").p.transform("max")
sel = sc[(sc.p >= best) & (sc.p >= 0.70)].drop_duplicates("pool_idx")
per = sel.groupby("s1_idx").size().reindex(range(len(s1)), fill_value=0)
top = sc.groupby("s1_idx").p.max().reindex(range(len(s1)), fill_value=0)
nc = sc.groupby("s1_idx").size().reindex(range(len(s1)), fill_value=0)
g = pd.DataFrame({"country": s1.country, "n_pred": per, "top_p": top, "n_cand": nc})
print(g.groupby("country").agg(n_pred_mean=("n_pred", "mean"), empty=("n_pred", lambda x: (x == 0).mean()),
      top_p_mean=("top_p", "mean"), n_cand=("n_cand", "mean")).round(3))
print("\nshare of pairs with 0.3<p<0.7 (uncertain), by country:")
print(sc.assign(u=(sc.p > 0.3) & (sc.p < 0.7)).groupby("country").u.mean().round(4))
print("\nn_pred distribution by country (train truth: 0:5.6% 1:5.4% 2:17% 3:24% 4:22% 5:15% 6+:11%):")
print(pd.crosstab(g.country, g.n_pred.clip(upper=6), normalize="index").round(3))

fr = g[g.country == "France"].sample(8, random_state=1).index
for i in fr:
    r = s1.iloc[i]; print(f"\n=== {r.business_name} | {r.business_address}")
    for _, c in sc[sc.s1_idx == i].nlargest(7, "p").iterrows():
        q = pool.iloc[int(c.pool_idx)]
        print(f"  p={c.p:.2f}  {q.business_name[:45]:45s} | {q.business_address[:60]}")
