"""BLOCK-DIAG: why are true pairs lost before the matcher? (full-scale, run on the T job's caches)

For every country: blocking recall, then for a sample of LOST true pairs (and found ones for comparison) rebuild
both records' keys, count each shared key's frequency over the FULL pool, and classify:
  no_shared_key   : no key in common -> normalization / transliteration / word-level mismatch
  only_capped     : keys in common but all above the df cap -> common names, chains, generic addresses
  ranked_out      : usable keys in common -> lost to competition in the wide cut or the pre-ranker
Extra flags per pair: empty pool address, native-script pool name, name / address token overlap, shared number.

  python scripts/diag_blocking.py --work <T work dir> --data <dataset dir> --out <dir> [--df-scale 1.0]"""
import argparse
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import src.blocking as blk
from src.prep import _read_arrow, read_tsv

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def pair_keys(c, n, a, u):
    out = {}
    for fam, ks in enumerate(blk.record_keys(c.lower(), n, a, u)):
        for k in ks:
            out[k] = fam + (4 if k[0] in blk.SINGLE_KINDS else 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--df-scale", type=float, default=1.0)
    ap.add_argument("--cap", type=int, default=1200)
    ap.add_argument("--cap-composite", type=int, default=600)
    ap.add_argument("--n-lost", type=int, default=40_000)
    ap.add_argument("--n-found", type=int, default=15_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    blk.MORE_KEYS = True                       # v4 runs use --extra-keys
    C = ["entity_id", "country", "n_clean", "a_clean", "nums"]
    s1 = _read_arrow(os.path.join(a.work, "train_norm_s1.parquet"))[C]
    pool = _read_arrow(os.path.join(a.work, "train_norm_pool.parquet"))[C]
    c = pd.read_parquet(os.path.join(a.work, "train_cands.parquet"), columns=["s1_idx", "pool_idx"])
    log(f"loaded s1 {len(s1):,} pool {len(pool):,} cands {len(c):,}")
    gt = read_tsv(os.path.join(a.data, "train", "train_ground_truth.tsv"))
    rows = [(s, m) for s, ms in zip(gt.source1_entity_id, gt.matched_entity_ids) for m in ms.split(",") if m]
    gi = pd.Index(s1.entity_id.values).get_indexer([r[0] for r in rows])
    gj = pd.Index(pool.entity_id.values).get_indexer([r[1] for r in rows])
    ok = (gi >= 0) & (gj >= 0)
    gi, gj = gi[ok], gj[ok]
    tk = (gi.astype(np.int64) << 24) | gj
    ck = np.unique((c.s1_idx.values.astype(np.int64) << 24) | c.pool_idx.values.astype(np.int64))
    found = np.isin(tk, ck)
    ct = np.asarray(s1.country.values, dtype=object)[gi]
    for cc in pd.unique(ct):
        m = ct == cc
        log(f"{cc}: true pairs {m.sum():,}  blocking recall {found[m].mean():.4f}  lost {int((~found[m]).sum()):,}")
    rng = np.random.default_rng(0)
    pick = []
    for cc in pd.unique(ct):
        for want, flag in ((a.n_lost, False), (a.n_found, True)):
            idx = np.flatnonzero((ct == cc) & (found == flag))
            pick.append(rng.choice(idx, min(want, len(idx)), replace=False))
    pick = np.concatenate(pick)
    P = pd.DataFrame({"s1_idx": gi[pick], "pool_idx": gj[pick], "found": found[pick], "country": ct[pick]})
    S1c = {k: np.asarray(s1[k].array.take(P.s1_idx.values), dtype=object) for k in C}
    PLc = {k: np.asarray(pool[k].array.take(P.pool_idx.values), dtype=object) for k in C}
    shared, need = [], set()
    for i in range(len(P)):
        ka = pair_keys(S1c["country"][i], S1c["n_clean"][i], S1c["a_clean"][i], S1c["nums"][i])
        kb = pair_keys(PLc["country"][i], PLc["n_clean"][i], PLc["a_clean"][i], PLc["nums"][i])
        sh = {k: f for k, f in ka.items() if k in kb}
        shared.append(sh)
        need.update(sh)
    log(f"{len(P):,} sampled pairs, {len(need):,} distinct shared keys; counting their pool frequency")
    need_l = list(need)
    need_h = pd.util.hash_array(np.array(need_l, dtype=object), categorize=False)
    order = np.argsort(need_h)
    need_hs = need_h[order]
    df = np.zeros(len(need_l), np.int64)
    cols = [pool.country.tolist(), pool.n_clean.tolist(), pool.a_clean.tolist(), pool.nums.tolist()]
    import multiprocessing as mp
    chunk = 200_000
    parts = [(i, *[x[i:i + chunk] for x in cols]) for i in range(0, len(pool), chunk)]
    del cols
    with mp.get_context("fork").Pool(os.cpu_count()) as pp:
        for r, h, f in pp.imap(blk._keys_chunk, parts, chunksize=1):
            m = np.isin(h, need_hs)
            if m.any():
                u, cnt = np.unique(h[m], return_counts=True)
                pos = np.searchsorted(need_hs, u)
                df[order[pos]] += cnt
    dfk = dict(zip(need_l, df))
    log("key frequencies counted")
    idf = lambda n: float(np.log((len(pool) + 1) / (n + 1)))
    out = []
    for i, sh in enumerate(shared):
        usable = {k: f for k, f in sh.items() if dfk[k] * a.df_scale <= (a.cap if f >= 4 else a.cap_composite)}
        fam = Counter((f % 4) for f in usable.values())
        cat = "no_shared_key" if not sh else ("only_capped" if not usable else "ranked_out")
        tn1, tn2 = set(S1c["n_clean"][i].split()), set(PLc["n_clean"][i].split())
        ta1, ta2 = set(S1c["a_clean"][i].split()), set(PLc["a_clean"][i].split())
        out.append(dict(cat=cat, n_shared=len(sh), n_usable=len(usable), usable_name=fam[0], usable_addr=fam[1],
                        usable_cross=fam[2], min_df_shared=min((dfk[k] for k in sh), default=-1),
                        usable_idf=sum(idf(dfk[k]) for k in usable),
                        name_j=len(tn1 & tn2) / max(len(tn1 | tn2), 1), addr_j=len(ta1 & ta2) / max(len(ta1 | ta2), 1),
                        num_shared=bool(set(S1c["nums"][i].split()) & set(PLc["nums"][i].split())),
                        pool_addr_empty=not PLc["a_clean"][i].strip(),
                        s1_n=S1c["n_clean"][i], s1_a=S1c["a_clean"][i], pool_n=PLc["n_clean"][i], pool_a=PLc["a_clean"][i],
                        s1_id=S1c["entity_id"][i], pool_id=PLc["entity_id"][i]))
    R = pd.concat([P.reset_index(drop=True), pd.DataFrame(out)], axis=1)
    raw = {}
    for k in (2, 3):
        t = read_tsv(os.path.join(a.data, "train", f"train_source{k}.tsv"))
        t = t[t.entity_id.isin(set(R.pool_id))]
        raw.update(zip(t.entity_id, t.business_name))
    R["pool_raw_name"] = R.pool_id.map(raw)
    R["native_script"] = R.pool_raw_name.map(lambda s: isinstance(s, str) and not s.isascii())
    R.to_parquet(os.path.join(a.out, "blocking_diag.parquet"))
    for cc in pd.unique(R.country):
        L = R[(R.country == cc) & ~R.found]
        Fd = R[(R.country == cc) & R.found]
        log(f"=== {cc}: lost sample {len(L):,} | category share: {L.cat.value_counts(normalize=True).round(3).to_dict()}")
        log(f"   lost: pool address empty {L.pool_addr_empty.mean():.3f} (found {Fd.pool_addr_empty.mean():.3f}) | native script "
            f"{L.native_script.mean():.3f} (found {Fd.native_script.mean():.3f}) | name jaccard median {L.name_j.median():.2f} "
            f"(found {Fd.name_j.median():.2f}) | addr jaccard median {L.addr_j.median():.2f} (found {Fd.addr_j.median():.2f})")
    log(f"wrote {a.out}/blocking_diag.parquet ({len(R):,} rows)")


if __name__ == "__main__":
    main()
