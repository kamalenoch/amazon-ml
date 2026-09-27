"""Final assembly (CPU): GBDT test scores + one or two cross-encoders + recall-supplement pairs -> one submission.

1. Holdout (labels): inside the CE band, p = w * p_gbdt + (1 - w) * (m * pc_A + (1 - m) * pc_B); grid over the GBDT
   weight w, the CE mix m (two CEs, e.g. e5-small and e5-base) and the threshold; best macro F0.5 wins
   (w = 1 is 'GBDT alone', so the choice can never be worse than the GBDT on the holdout).
2. Test: the same blend on the band rows of the merged GBDT test scores, best S1 per pool record, threshold.
3. Recall supplement: pairs with pc >= its calibrated threshold whose pool record is still unmatched and that are
   not candidates of this run (those were scored here) are added.
4. matching_results.tsv + candidate_pairs.tsv (candidates + added pairs).

  python scripts/final_assemble.py --data <dataset> --train-work <T dir> --scores <F dir> [<F dir>]
      --ce <CE out dir> [<CE out dir>] [--supplement <RS out dir>] --out <dir>
CE out dirs need ce_holdout_band.parquet and ce_test_band.parquet (rows index holdout_preds / the test scores)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.run import s1_hash, write_lists

T0 = time.time()
KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def best_rows(pool, p):
    o = np.lexsort((-p, pool))
    ps = pool[o]
    return o[np.r_[True, ps[1:] != ps[:-1]]] if len(o) else o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-work", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--ce", nargs="*", default=[])
    ap.add_argument("--supplement", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", default=None, help="testing: 'w,thr[,mix]' instead of the holdout choice")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---------------- holdout
    rep = json.load(open(os.path.join(a.train_work, "train_report.json")))["args"]
    hp = pd.read_parquet(os.path.join(a.train_work, "holdout_preds.parquet"))
    tr1 = pd.read_csv(f"{a.data}/train/train_source1.tsv", usecols=["entity_id"], **KW).entity_id.values
    gt = pd.read_csv(f"{a.data}/train/train_ground_truth.tsv", **KW)
    T = pd.Series(gt.matched_entity_ids.str.count("S[23]-").values, gt.source1_entity_id.values).reindex(tr1).fillna(0).values
    h = s1_hash(tr1)
    pr_end = int(rep["pr_frac"] * 10_000); m_end = pr_end + int(rep["m_frac"] * 10_000)
    hcut = pr_end + (1 - rep["holdout"]) * (m_end - pr_end)
    hs1 = np.flatnonzero((h >= hcut) & (h < m_end))
    n = len(tr1)
    hs, hj, hy, hpg = hp.s1_idx.values, hp.pool_idx.values, hp.y.values.astype(float), hp.p.values.astype(np.float32)
    ce_h, ce_t = [], []
    for d in a.ce:
        hb = pd.read_parquet(os.path.join(d, "ce_holdout_band.parquet"))
        tb = pd.read_parquet(os.path.join(d, "ce_test_band.parquet"))
        band_here = np.flatnonzero((hp.p.values >= 0.01) & (hp.p.values <= 0.99))
        assert len(band_here) == len(hb) and (band_here == np.sort(hb.row.values)).all(), \
            f"{d}: its holdout rows are not this holdout's band -> that CE ran on another GBDT (attach the CE run of THIS version)"
        ce_h.append(hb); ce_t.append(tb)
        log(f"CE {d}: holdout band {len(hb):,} rows, test band {len(tb):,} rows (holdout rows match this T)")
    if len(ce_h) == 2:
        assert (ce_h[0].row.values == ce_h[1].row.values).all() and (ce_t[0].row.values == ce_t[1].row.values).all(), \
            "the two CE runs must re-score the same band rows (same GBDT scores)"
    thrs = np.round(np.arange(0.40, 0.951, 0.025), 3)

    def grid_eval(p):
        b = best_rows(hj, p)
        s, q, y = hs[b], p[b], hy[b]
        out = []
        for t in thrs:
            m = q >= t
            tp = np.bincount(s[m], weights=y[m], minlength=n); npr = np.bincount(s[m], minlength=n)
            f = np.where(npr > 0, 1.25 * tp / np.maximum(npr + 0.25 * T, 1e-9), (T == 0).astype(float))
            out.append((f[hs1].mean(), t))
        return max(out)

    best = (*grid_eval(hpg), 1.0, None)                                     # (F0.5, thr, w, mix)
    log(f"holdout GBDT alone: F0.5 {best[0]:.4f} @ {best[1]}")
    if ce_h:
        row = ce_h[0].row.values
        mixes = [None] if len(ce_h) == 1 else [0.0, 0.25, 0.5, 0.75, 1.0]
        for mix in mixes:
            pc = ce_h[0].pc.values.astype(np.float32) if mix is None else \
                (mix * ce_h[0].pc.values + (1 - mix) * ce_h[1].pc.values).astype(np.float32)
            for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                p = hpg.copy()
                p[row] = w * hpg[row] + (1 - w) * pc
                f, t = grid_eval(p)
                if f > best[0]:
                    best = (f, t, w, mix)
            log(f"  mix {mix}: best so far F0.5 {best[0]:.4f} (thr {best[1]}, GBDT weight {best[2]}, mix {best[3]})")
    f_ho, thr, w, mix = best
    if a.force:
        v = [float(x) for x in a.force.split(",")]
        w, thr, mix = v[0], v[1], (v[2] if len(v) > 2 else mix)
        log(f"FORCED: GBDT weight {w}, threshold {thr}, mix {mix}")
    log(f"CHOSEN: holdout F0.5 {f_ho:.4f} | GBDT weight {w} | CE mix {mix} (first CE share) | threshold {thr}")
    del hp, hs, hj, hy, hpg

    # ---------------- test
    ts = None
    for d in a.scores:
        x = pd.read_parquet(os.path.join(d, "test_scores.parquet"))
        if ts is None:
            ts = x
        else:
            assert len(x) == len(ts) and (x.s1_idx.values == ts.s1_idx.values).all(), "different candidate sets"
            ts["p"] = np.maximum(ts.p.values, x.p.values)
    ts1, tpj, p = ts.s1_idx.values, ts.pool_idx.values, ts.p.values.astype(np.float32)
    del ts
    for d in a.ce:                                    # the CE must have re-scored THESE test pairs
        fb = os.path.join(d, "test_scores_blend.parquet")
        if os.path.exists(fb):
            x = pd.read_parquet(fb, columns=["s1_idx", "pool_idx"])
            assert len(x) == len(ts1) and (x.s1_idx.values == ts1).all() and (x.pool_idx.values == tpj).all(), \
                f"{d}: its test pairs differ from these scores -> attach the CE run of THIS version"
            del x
            log(f"CE {d}: test pairs identical to the scores")
    if ce_t and w < 1.0:
        row = ce_t[0].row.values
        pc = ce_t[0].pc.values.astype(np.float32) if mix is None else \
            (mix * ce_t[0].pc.values + (1 - mix) * ce_t[1].pc.values).astype(np.float32)
        p[row] = w * p[row] + (1 - w) * pc
    b = best_rows(tpj, p)
    b = b[p[b] >= thr]
    S, P = ts1[b], tpj[b]
    log(f"test: {len(p):,} candidate pairs -> {len(S):,} matched")
    te = f"{a.data}/test"
    t1 = pd.read_csv(f"{te}/test_source1.tsv", usecols=["entity_id", "country"], **KW)
    pool_ids = pd.concat([pd.read_csv(f"{te}/test_source{k}.tsv", usecols=["entity_id"], **KW) for k in (2, 3)]).entity_id.values
    add_s = add_p = np.zeros(0, np.int64)
    if a.supplement:
        cal = json.load(open(os.path.join(a.supplement, "supplement_calibration.json")))
        sp = pd.read_parquet(os.path.join(a.supplement, "supplement_pairs.parquet"))
        t_use = cal.get("t_use")
        if t_use is None:
            log("supplement: no holdout gain in its calibration -> nothing added")
        else:
            sp = sp[sp.pc >= t_use]
            si = pd.Index(t1.entity_id.values).get_indexer(sp.s1_id.values)
            pj = pd.Index(pool_ids).get_indexer(sp.pool_id.values)
            ok = (si >= 0) & (pj >= 0)
            si, pj = si[ok].astype(np.int64), pj[ok].astype(np.int64)
            matched = np.zeros(len(pool_ids), bool); matched[P] = True
            key = np.sort((ts1.astype(np.int64) << 24) | tpj.astype(np.int64))
            k = (si << 24) | pj
            pos = np.clip(np.searchsorted(key, k), 0, len(key) - 1)
            is_cand = key[pos] == k
            keep = ~matched[pj] & ~is_cand
            add_s, add_p = si[keep], pj[keep]
            log(f"supplement: {len(sp):,} pairs >= {t_use} -> {int(keep.sum()):,} added "
                f"(dropped: {int(matched[pj].sum()):,} pool already matched, {int(is_cand.sum()):,} already candidates) | by country "
                f"{pd.Series(t1.country.values[add_s]).value_counts().to_dict()}")
    S = np.concatenate([S, add_s]); P = np.concatenate([P, add_p])
    n_has = write_lists(os.path.join(a.out, "matching_results.tsv"), t1.entity_id.values, S, pd.array(pool_ids, dtype="string"), P,
                        "matched_entity_ids")
    write_lists(os.path.join(a.out, "candidate_pairs.tsv"), t1.entity_id.values, np.concatenate([ts1, add_s]),
                pd.array(pool_ids, dtype="string"), np.concatenate([tpj, add_p]), "candidate_entity_ids")
    json.dump(dict(holdout_f05=f_ho, thr=thr, gbdt_weight=w, ce_mix=mix, added=int(len(add_s))), open(os.path.join(a.out, "final_choice.json"), "w"))
    log(f"DONE: {len(S):,} matched pairs ({len(S) / len(t1):.2f} per S1), {n_has.mean():.3f} of S1 with a match; "
        f"by country {pd.Series(n_has).groupby(t1.country.values).mean().round(3).to_dict()}")


if __name__ == "__main__":
    main()
