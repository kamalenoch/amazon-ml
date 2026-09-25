"""Build the regional dev subset (all train records of 5 states: OR, KY, UT, Kerala, Rajasthan).
Keeps realistic address-key density. Usage: python scripts/make_dev_subset.py <train_dir> <out_dir>
Pool = all S2/S3 matched to subset S1 + every S2/S3 record whose address mentions those states."""
import os, sys
import pandas as pd

D, O = sys.argv[1], sys.argv[2]
os.makedirs(O, exist_ok=True)
ST = {"or", "oregon", "ky", "kentucky", "ut", "utah", "kerala", "kl", "rajasthan", "rj"}
kw = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3)


def hit(a):
    return any(p.strip().lower() in ST for p in a.split(","))


s1 = pd.read_csv(os.path.join(D, "train_source1.tsv"), **kw)
s = s1[s1.business_address.map(hit)]
s.to_csv(os.path.join(O, "train_source1.tsv"), sep="\t", index=False)
gt = pd.read_csv(os.path.join(D, "train_ground_truth.tsv"), **kw)
g = gt[gt.source1_entity_id.isin(set(s.entity_id))]
g.to_csv(os.path.join(O, "train_ground_truth.tsv"), sep="\t", index=False)
m = {i for x in g.matched_entity_ids for i in x.split(",") if i}
for k in (2, 3):
    out = [ch[ch.entity_id.isin(m) | ch.business_address.map(hit)]
           for ch in pd.read_csv(os.path.join(D, f"train_source{k}.tsv"), chunksize=500_000, **kw)]
    pd.concat(out).to_csv(os.path.join(O, f"train_source{k}.tsv"), sep="\t", index=False)
print("done:", len(s), "S1")
