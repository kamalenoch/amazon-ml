"""Stitch per-country predict outputs and re-select with thresholds adjusted for the test decoy rate.

Test has ~5.8 pool records per US/India S1 vs ~4.7 in train; the extra records are negatives that land in the
mid-probability band, so the holdout-tuned threshold is too permissive on test. Per country:
  m_b = (test pairs per S1 in p-bin b - holdout positives per S1 in b) / holdout negatives per S1 in b
(best-S1-per-pool pairs only), then the threshold that maximizes holdout macro F0.5 with every holdout negative
in bin b counted m_b times. France (no labels) uses the pooled US+India holdout as reference.
Writes two submissions: <out>/base (the model's own selection) and <out>/shift (per-country adjusted thresholds).

  python scripts/reselect.py --scores <F-job work dirs...> --train-work <T or T2 work dir> --test-work <S work dir>
      --gt <data>/train/train_ground_truth.tsv --cands <one F-job output dir> --out <dir>"""
import argparse
import os
import pickle
import shutil

import numpy as np
import pandas as pd

BINS = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.001])
THRS = np.round(np.arange(0.50, 0.951, 0.025), 3)


def s1_hash(ids):
    return (pd.util.hash_array(np.asarray(ids, dtype=object)) % 10_000).astype(np.int32)


def best_per_pool(s1, pool, p):
    d = pd.DataFrame({"s1": s1, "pool": pool, "p": p})
    d = d[d.p > 0]
    d = d.sort_values("p", ascending=False, kind="stable").drop_duplicates("pool")
    return d


def per_s1_bins(d, n_s1, col=None):
    """pairs per S1 in each p bin (optionally only rows where col is True/False)."""
    b = np.digitize(d.p.values, BINS) - 1
    ok = (b >= 0) & (b < len(BINS) - 1)
    return np.bincount(b[ok], minlength=len(BINS) - 1) / max(n_s1, 1)


def weighted_f05(h, k_true, s1_all, thr, m):
    """holdout macro F0.5 when each negative in bin b counts m[b] times."""
    sel = h[h.p >= thr]
    b = np.clip(np.digitize(sel.p.values, BINS) - 1, 0, len(BINS) - 2)
    w = np.where(sel.y.values > 0, 0.0, m[b])
    tp = pd.Series(sel.y.values.astype(float)).groupby(sel.s1.values).sum()
    fp = pd.Series(w).groupby(sel.s1.values).sum()
    k = k_true.reindex(s1_all).fillna(0).values
    tpv = tp.reindex(s1_all).fillna(0).values
    fpv = fp.reindex(s1_all).fillna(0).values
    npred = tpv + fpv
    f = np.where(npred == 0, np.where(k == 0, 1.0, 0.0), 1.25 * tpv / np.maximum(tpv + fpv + 0.25 * k, 1e-9))
    f = np.where((k == 0) & (npred > 0), 0.0, f)
    return f.mean()


def write_lists(path, s1_ids, s1_idx, pool_ids, pool_idx, col):
    order = np.lexsort((pool_idx, s1_idx))
    s1_idx, pool_idx = s1_idx[order], pool_idx[order]
    bounds = np.searchsorted(s1_idx, np.arange(len(s1_ids) + 1))
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for r in range(len(s1_ids)):
            f.write(f"{s1_ids[r]}\t{','.join(pool_ids[pool_idx[bounds[r]:bounds[r + 1]]])}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="work dirs of the predict jobs (test_scores.parquet)")
    ap.add_argument("--train-work", required=True, help="dir with holdout_preds.parquet + model.pkl (+ train_norm_s1)")
    ap.add_argument("--train-s1", default=None, help="train_norm_s1.parquet if not in --train-work")
    ap.add_argument("--test-work", required=True, help="dir with test_norm_s1.parquet + test_norm_pool.parquet")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--cands", required=True, help="dir with candidate_pairs.tsv (copied to both submissions)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-raise", type=float, default=0.15)
    a = ap.parse_args()

    # merged test scores (each predict job holds p for its countries, 0 elsewhere; same candidate order)
    sc = None
    for d in a.scores:
        x = pd.read_parquet(os.path.join(d, "test_scores.parquet"))
        if sc is None:
            sc = x
        else:
            assert len(x) == len(sc) and (x.s1_idx.values == sc.s1_idx.values).all(), "different candidate sets"
            sc["p"] = np.maximum(sc.p.values, x.p.values)
    t1 = pd.read_parquet(os.path.join(a.test_work, "test_norm_s1.parquet"), columns=["entity_id", "country"])
    tp_ids = np.asarray(pd.read_parquet(os.path.join(a.test_work, "test_norm_pool.parquet"), columns=["entity_id"]).entity_id.values, dtype=object)
    t_ids = np.asarray(t1.entity_id.values, dtype=object)
    t_ct = np.asarray(t1.country.values, dtype=object)
    M = pickle.load(open(os.path.join(a.train_work, "model.pkl"), "rb"))
    method = M.get("method", ("threshold", M["thr"]))
    base_thr = round(float(M["thr"]), 3)
    print("model selection:", method, "| holdout threshold", base_thr)

    # holdout with labels and S1 countries
    hp = pd.read_parquet(os.path.join(a.train_work, "holdout_preds.parquet"))
    tr_path = a.train_s1 or os.path.join(a.train_work, "train_norm_s1.parquet")
    tr = pd.read_parquet(tr_path, columns=["entity_id", "country"])
    tr_ids, tr_ct = np.asarray(tr.entity_id.values, dtype=object), np.asarray(tr.country.values, dtype=object)
    gt = pd.read_csv(a.gt, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    k_all = pd.Series(gt.matched_entity_ids.str.count("S[23]-").values, gt.source1_entity_id.values)
    hs1 = np.unique(hp.s1_idx.values)
    h = best_per_pool(hp.s1_idx.values, hp.pool_idx.values, hp.p.values).rename(columns={})
    h["y"] = hp.set_index(["s1_idx", "pool_idx"]).y.reindex(pd.MultiIndex.from_arrays([h.s1.values, h.pool.values])).values
    h_ct = tr_ct[h.s1.values]
    k_true = pd.Series(k_all.reindex(tr_ids[hs1]).fillna(0).values, index=hs1)

    t = best_per_pool(sc.s1_idx.values, sc.pool_idx.values, sc.p.values)
    t_ctry = t_ct[t.s1.values]
    thr_by = {}
    for ct in pd.unique(t_ct):
        ref = [c for c in pd.unique(tr_ct) if c == ct] or list(pd.unique(tr_ct))     # unseen country: pooled holdout
        hm = np.isin(h_ct, ref)
        hs = hs1[np.isin(tr_ct[hs1], ref)]
        hh = h[hm]
        pos = per_s1_bins(hh[hh.y > 0], len(hs))
        neg = per_s1_bins(hh[hh.y == 0], len(hs))
        n_neg = np.bincount(np.clip(np.digitize(hh[hh.y == 0].p.values, BINS) - 1, 0, len(BINS) - 2), minlength=len(BINS) - 1)
        n_t = int((t_ct == ct).sum())
        tot = per_s1_bins(t[t_ctry == ct], n_t)
        m = np.where(n_neg >= 20, np.clip((tot - pos) / np.maximum(neg, 1e-9), 0.5, 5.0), 1.0)
        f_plain = {thr: weighted_f05(hh, k_true, hs, thr, np.ones_like(m)) for thr in THRS}
        f_shift = {thr: weighted_f05(hh, k_true, hs, thr, m) for thr in THRS}
        t_plain = max(f_plain, key=f_plain.get)
        t_shift = max(f_shift, key=f_shift.get)
        t_use = min(max(t_shift, base_thr), base_thr + a.max_raise)
        thr_by[ct] = t_use
        print(f"{ct}: {n_t:,} S1 | ref {ref} | decoy multiplier by bin {np.round(m, 2).tolist()}")
        print(f"   holdout-best thr {t_plain} | shift-aware thr {t_shift} (weighted F0.5 {f_shift[t_shift]:.4f} vs "
              f"{f_shift.get(base_thr, float('nan')):.4f} at {base_thr}) -> using {t_use}")

    for name, rule in (("base", None), ("shift", thr_by)):
        od = os.path.join(a.out, name)
        os.makedirs(od, exist_ok=True)
        if rule is None and method[0] == "expected_f":
            import sys
            sys.path.insert(0, os.getcwd())
            from src.select import expected_f_select
            sel = expected_f_select(sc.s1_idx.values, sc.pool_idx.values, sc.p.values, miss_prior=method[1][0], power=method[1][1])
            s_s1, s_pool = sel.s1.values, sel.pool.values
        else:
            thr_row = np.full(len(t), float(method[1]) if rule is None else 0.0)
            if rule is not None:
                thr_row = np.array([rule[c] for c in t_ctry])
            keep = t.p.values >= thr_row
            s_s1, s_pool = t.s1.values[keep], t.pool.values[keep]
        write_lists(os.path.join(od, "matching_results.tsv"), t_ids, np.asarray(s_s1, np.int64), tp_ids,
                    np.asarray(s_pool, np.int64), "matched_entity_ids")
        shutil.copy(os.path.join(a.cands, "candidate_pairs.tsv"), os.path.join(od, "candidate_pairs.tsv"))
        cnt = pd.Series(t_ct[np.asarray(s_s1, np.int64)]).value_counts() / pd.Series(t_ct).value_counts()
        print(f"{name}: {len(s_s1):,} matched pairs; per S1 by country {cnt.round(3).to_dict()}")


if __name__ == "__main__":
    main()
