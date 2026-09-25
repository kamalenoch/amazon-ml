"""End-to-end pipeline: data -> normalize -> blocking -> features -> LightGBM -> selection -> TSVs.

python -m src.run --data <.../student_resource/dataset> --work <cache dir> --out <output dir>

Every stage caches to --work, so a re-run skips finished stages (delete a file to redo it).
"""
import argparse
import gc
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from .prep import load_split, load_gt
from .blocking import generate_candidates
from .features import build_idf, cheap_features, string_features
from .matching import train_lgb, feat_cols, select, f05_macro

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def cached(path, fn):
    if os.path.exists(path):
        log("cache hit", os.path.basename(path))
        return pd.read_parquet(path)
    df = fn()
    df.to_parquet(path)
    return df


def s1_hash(ids):
    return (pd.util.hash_array(np.asarray(ids, dtype=object)) % 10_000).astype(np.int32)


def labels(c, s1, pool, gtp):
    k = pd.MultiIndex.from_arrays([s1.entity_id.values[c.s1_idx.values], pool.entity_id.values[c.pool_idx.values]])
    return k.isin(pd.MultiIndex.from_arrays([gtp.s1_id.values, gtp.pool_id.values])).astype(np.int8)


def write_lists(path, s1_ids, pairs_s1, pairs_pool, col):
    """One row per S1 (empty list allowed), ids comma-joined, no duplicates."""
    d = pd.DataFrame({"s1": pairs_s1, "pool": pairs_pool}).drop_duplicates()
    lists = d.groupby("s1").pool.agg(",".join)
    out = pd.DataFrame({"source1_entity_id": s1_ids})
    out[col] = out.source1_entity_id.map(lists).fillna("")
    out.to_csv(path, sep="\t", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="output")
    ap.add_argument("--train-frac", type=float, default=0.30, help="fraction of train S1 used to fit the model")
    ap.add_argument("--holdout", type=float, default=0.2, help="share of that sample held out for threshold tuning")
    ap.add_argument("--cap", type=int, default=600)
    ap.add_argument("--df-scale", type=float, default=1.0)
    ap.add_argument("--k-total", type=int, default=30)
    ap.add_argument("--k-name", type=int, default=10)
    ap.add_argument("--k-addr", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--threshold", type=float, default=None, help="override the tuned threshold")
    ap.add_argument("--predict-chunk", type=int, default=4_000_000)
    ap.add_argument("--jobs", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    W = lambda f: os.path.join(a.work, f)
    bkw = dict(cap=a.cap, df_scale=a.df_scale, k_total=a.k_total, k_name=a.k_name, k_addr=a.k_addr, n_jobs=a.jobs)

    # ------------------------------------------------------------------ TRAIN
    model_path = W("model.pkl")
    if not os.path.exists(model_path):
        s1, pool = load_split(a.data, "train", a.work, a.jobs)
        log("train loaded", len(s1), len(pool))
        _, gtp = load_gt(os.path.join(a.data, "train", "train_ground_truth.tsv"))
        c = cached(W("train_cands.parquet"), lambda: generate_candidates(s1, pool, **bkw))
        y_all = labels(c, s1, pool, gtp)
        log(f"train cands {len(c):,}  per S1 {len(c) / len(s1):.1f}  "
            f"blocking recall {y_all.sum() / len(gtp):.4f}")
        # competition context needs the FULL candidate set -> compute before sampling
        cheap = cheap_features(c, s1, pool)
        h = s1_hash(s1.entity_id.values)[c.s1_idx.values]
        use = h < a.train_frac * 10_000
        c, cheap, y = c[use].reset_index(drop=True), cheap[use].reset_index(drop=True), y_all[use]
        h = h[use]
        del y_all
        gc.collect()
        idf_n = build_idf([s1.n_clean, pool.n_clean])
        idf_a = build_idf([s1.a_clean, pool.a_clean])
        log(f"string features on {len(c):,} sampled pairs")
        X = pd.concat([string_features(c, s1, pool, idf_n, idf_a, a.jobs), cheap], axis=1)
        del cheap
        F = feat_cols(X)
        va = h >= (a.train_frac * (1 - a.holdout)) * 10_000
        log(f"fit lgb: train {(~va).sum():,} rows, holdout {va.sum():,} rows, pos rate {y.mean():.3f}")
        m = train_lgb(X.loc[~va, F], y[~va], rounds=a.rounds, valid=(X.loc[va, F], y[va]))
        p = m.predict(X.loc[va, F], num_threads=0)
        # threshold tuning on holdout S1s against the FULL ground truth (blocking misses count)
        hs1 = np.unique(c.s1_idx.values[va])
        ids = s1.entity_id.values
        true = gtp[gtp.s1_id.isin(set(ids[hs1]))].rename(columns={"s1_id": "s1", "pool_id": "pool"})
        best = (-1, 0.5)
        for thr in np.arange(0.30, 0.92, 0.04):
            sel = select(c.s1_idx.values[va], c.pool_idx.values[va], p, thr)
            pred = pd.DataFrame({"s1": ids[sel.s1.values], "pool": pool.entity_id.values[sel.pool.values]})
            f, info = f05_macro(pred, true, ids[hs1])
            log(f"  thr {thr:.2f}  F0.5 {f:.4f}  {info}")
            best = max(best, (f, thr))
        log(f"HOLDOUT F0.5 {best[0]:.4f} at threshold {best[1]:.2f}")
        imp = pd.Series(m.feature_importance("gain"), F).sort_values(ascending=False)
        log("top features", imp.head(15).round(0).to_dict())
        with open(model_path, "wb") as f:
            pickle.dump(dict(model=m, feats=F, thr=float(best[1]), holdout_f05=best[0]), f)
        json.dump(dict(holdout_f05=best[0], threshold=float(best[1]), best_iter=m.best_iteration,
                       features=F), open(W("train_report.json"), "w"), indent=1)
        del X, c, s1, pool
        gc.collect()

    with open(model_path, "rb") as f:
        M = pickle.load(f)
    thr = a.threshold if a.threshold is not None else M["thr"]
    log(f"using threshold {thr:.2f}")

    # ------------------------------------------------------------------ TEST
    s1, pool = load_split(a.data, "test", a.work, a.jobs)
    log("test loaded", len(s1), len(pool), s1.country.value_counts().to_dict())
    c = cached(W("test_cands.parquet"), lambda: generate_candidates(s1, pool, **bkw))
    log(f"test cands {len(c):,}  per S1 {len(c) / len(s1):.1f}")
    write_lists(os.path.join(a.out, "candidate_pairs.tsv"), s1.entity_id.values,
                s1.entity_id.values[c.s1_idx.values], pool.entity_id.values[c.pool_idx.values],
                "candidate_entity_ids")
    cheap = cheap_features(c, s1, pool)
    idf_n = build_idf([s1.n_clean, pool.n_clean])
    idf_a = build_idf([s1.a_clean, pool.a_clean])
    p = np.zeros(len(c), np.float32)
    for k in range(0, len(c), a.predict_chunk):
        sl = slice(k, k + a.predict_chunk)
        cc = c.iloc[sl]
        X = pd.concat([string_features(cc, s1, pool, idf_n, idf_a, a.jobs), cheap.iloc[sl]], axis=1)
        p[sl] = M["model"].predict(X[M["feats"]], num_threads=0)
        log(f"  predicted {min(k + a.predict_chunk, len(c)):,}/{len(c):,}")
    pd.DataFrame({"s1_idx": c.s1_idx.values, "pool_idx": c.pool_idx.values, "p": p}).to_parquet(W("test_scores.parquet"))
    sel = select(c.s1_idx.values, c.pool_idx.values, p, thr)
    out = write_lists(os.path.join(a.out, "matching_results.tsv"), s1.entity_id.values,
                      s1.entity_id.values[sel.s1.values], pool.entity_id.values[sel.pool.values],
                      "matched_entity_ids")
    n = out.matched_entity_ids.str.len().gt(0)
    log(f"DONE: {len(out):,} S1 rows, {n.mean():.3f} with >=1 match, {len(sel):,} matched pairs")
    log("by country (share with match):", out.assign(c=s1.country.values, n=n).groupby("c").n.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
