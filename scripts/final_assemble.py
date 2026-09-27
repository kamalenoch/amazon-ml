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
import re
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


_TOK = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"\d+")
_NL = re.compile(r"[^\x00-\x7F]")
STACK_TEXT = ["pool_name_nonlatin", "s1_name_nonlatin", "pool_addr_empty", "house_number_rel", "name_jacc", "addr_jacc", "name_len_ratio"]


def pair_text_feats(n1, a1, n2, a2):
    """Cheap raw-text flags for the learned combiner (script, empty address, house number, token overlap)."""
    out = np.empty((len(n1), len(STACK_TEXT)), np.float32)
    for i, (x1, y1, x2, y2) in enumerate(zip(n1, a1, n2, a2)):
        t1, t2 = set(_TOK.findall(x1.lower())), set(_TOK.findall(x2.lower()))
        u1, u2 = set(_TOK.findall(y1.lower())), set(_TOK.findall(y2.lower()))
        m1, m2 = _NUM.search(y1), _NUM.search(y2)
        if m1 and m2:
            d = abs(int(m1.group(0)[:9]) - int(m2.group(0)[:9]))
            hn = 0.0 if d == 0 else (1.0 if d <= 20 else 2.0)
        else:
            hn = -1.0
        out[i] = (bool(_NL.search(x2)), bool(_NL.search(x1)), not y2.strip(), hn,
                  len(t1 & t2) / max(len(t1 | t2), 1), len(u1 & u2) / max(len(u1 | u2), 1), len(x2) / max(len(x1), 1))
    return out


def read_texts(paths):
    df = pd.concat([pd.read_csv(p, usecols=["business_name", "business_address"], **KW) for p in paths], ignore_index=True)
    return df.business_name.values, df.business_address.values


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
    ap.add_argument("--llm", default=None, help="LLM re-scorer out dir (llm_holdout_band / llm_test_band: rows it scored)")
    ap.add_argument("--fr-variants", type=float, default=0.0, help="also write variants/fr_plus, variants/fr_minus with the France threshold +/- this delta (untested on the leaderboard: a gamble for spare submissions)")
    ap.add_argument("--stack", action="store_true", help="try a learned combiner (LightGBM, 2-fold CV on the holdout band); used only if it beats the weighted blend by >= 0.0003")
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
    # ---- optional LLM on the rows it scored: p = (1 - v) * p_blend + v * p_llm; v and threshold on the holdout
    v_llm = 0.0
    if a.llm:
        lh = pd.read_parquet(os.path.join(a.llm, "llm_holdout_band.parquet"))
        lh = lh[np.isfinite(lh.pc.values)]
        assert len(lh) == 0 or lh.row.max() < len(hpg), "LLM holdout rows do not index this holdout"
        pb = hpg.copy()
        if ce_h and w < 1.0:
            row = ce_h[0].row.values
            pcb = ce_h[0].pc.values.astype(np.float32) if mix is None else \
                (mix * ce_h[0].pc.values + (1 - mix) * ce_h[1].pc.values).astype(np.float32)
            pb[row] = w * hpg[row] + (1 - w) * pcb
        f0, t0 = grid_eval(pb)
        best_l = (f0, t0, 0.0)
        lr_, lp_ = lh.row.values, lh.pc.values.astype(np.float32)
        for v in (0.25, 0.5, 0.75, 1.0):
            p2 = pb.copy()
            p2[lr_] = (1 - v) * pb[lr_] + v * lp_
            f, t = grid_eval(p2)
            log(f"  LLM share {v}: holdout F0.5 {f:.4f} @ {t}")
            if f > best_l[0] + (2e-4 if best_l[2] == 0.0 else 0.0):
                best_l = (f, t, v)
        if best_l[2] > 0:
            f_ho, thr, v_llm = best_l
        log(f"LLM ({len(lh):,} holdout rows): {'USED, share ' + str(v_llm) if v_llm else 'not used (no holdout gain >= 0.0002)'}"
            f" | holdout F0.5 {best_l[0]:.4f} vs {f0:.4f} without")
    log(f"CHOSEN: holdout F0.5 {f_ho:.4f} | GBDT weight {w} | CE mix {mix} (first CE share) | LLM share {v_llm} | threshold {thr}")
    stack_model = None
    if a.stack and ce_h:
        try:
            import lightgbm as lgb
            brow = ce_h[0].row.values
            lmap = None
            if a.llm:
                lh_ = pd.read_parquet(os.path.join(a.llm, "llm_holdout_band.parquet"))
                lmap = pd.Series(lh_.pc.values, lh_.row.values)
            def stack_X(p_g, pcs, pl, tf):
                cols = [p_g] + pcs + ([pl] if pl is not None else [])
                return np.column_stack(cols + [tf]).astype(np.float32)
            s1n, s1a = read_texts([f"{a.data}/train/train_source1.tsv"])
            pn, pa = read_texts([f"{a.data}/train/train_source{k}.tsv" for k in (2, 3)])
            tf_h = pair_text_feats(s1n[hs[brow]], s1a[hs[brow]], pn[hj[brow]], pa[hj[brow]])
            del s1n, s1a, pn, pa
            pl_h = lmap.reindex(brow).values.astype(np.float32) if lmap is not None else None
            Xh = stack_X(hpg[brow], [c.pc.values.astype(np.float32) for c in ce_h], pl_h, tf_h)
            yh = hy[brow]
            fold = (s1_hash(tr1[hs[brow]]) % 2).astype(int)
            prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.9,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, seed=7)
            oof = np.empty(len(brow), np.float32)
            for k in (0, 1):
                mdl = lgb.train(prm, lgb.Dataset(Xh[fold != k], yh[fold != k]), 300)
                oof[fold == k] = mdl.predict(Xh[fold == k])
            ps = hpg.copy()
            ps[brow] = oof
            f_st, t_st = grid_eval(ps)
            log(f"learned combiner (2-fold OOF on {len(brow):,} band rows, {Xh.shape[1]} features): holdout F0.5 {f_st:.4f} @ {t_st} "
                f"vs weighted blend {f_ho:.4f}")
            if f_st > f_ho + 3e-4 or os.environ.get("STACK_FORCE"):          # STACK_FORCE: testing only
                stack_model = lgb.train(prm, lgb.Dataset(Xh, yh), 300)
                f_ho, thr = f_st, t_st
                log(f"CHOSEN (learned combiner): holdout F0.5 {f_ho:.4f} | threshold {thr}")
            else:
                log("learned combiner not used (gain < 0.0003)")
        except Exception as e:                                   # never lose the final over the optional combiner
            log(f"learned combiner failed ({type(e).__name__}: {e}) -> weighted blend")
            stack_model = None
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
    p_raw = p.copy() if stack_model is not None else None
    if stack_model is not None:
        te_ = f"{a.data}/test"
        brow = ce_t[0].row.values
        s1n, s1a = read_texts([f"{te_}/test_source1.tsv"])
        pn, pa = read_texts([f"{te_}/test_source{k}.tsv" for k in (2, 3)])
        tf_t = pair_text_feats(s1n[ts1[brow]], s1a[ts1[brow]], pn[tpj[brow]], pa[tpj[brow]])
        del s1n, s1a, pn, pa
        pl_t = None
        if a.llm:
            lt_ = pd.read_parquet(os.path.join(a.llm, "llm_test_band.parquet"))
            pl_t = pd.Series(lt_.pc.values, lt_.row.values).reindex(brow).values.astype(np.float32)
        Xt = np.column_stack([p_raw[brow]] + [c.pc.values.astype(np.float32) for c in ce_t] + ([pl_t] if pl_t is not None else []) + [tf_t]).astype(np.float32)
        p[brow] = stack_model.predict(Xt).astype(np.float32)
        log(f"learned combiner applied to {len(brow):,} test band rows")
        ce_t_apply = False
    else:
        ce_t_apply = True
    if ce_t and w < 1.0 and ce_t_apply:
        row = ce_t[0].row.values
        pc = ce_t[0].pc.values.astype(np.float32) if mix is None else \
            (mix * ce_t[0].pc.values + (1 - mix) * ce_t[1].pc.values).astype(np.float32)
        p[row] = w * p[row] + (1 - w) * pc
    if v_llm > 0 and stack_model is None:
        lt = pd.read_parquet(os.path.join(a.llm, "llm_test_band.parquet"))
        lt = lt[np.isfinite(lt.pc.values)]
        assert len(lt) == 0 or lt.row.max() < len(p), "LLM test rows do not index these scores"
        p[lt.row.values] = (1 - v_llm) * p[lt.row.values] + v_llm * lt.pc.values.astype(np.float32)
        log(f"LLM applied to {len(lt):,} test rows (share {v_llm})")
    b_all = best_rows(tpj, p)
    b = b_all[p[b_all] >= thr]
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
    if a.fr_variants > 0:                       # France threshold variants: same scores + same supplement, only France rows differ
        is_fr = (t1.country.values == "France")
        for tag, d in (("fr_plus", a.fr_variants), ("fr_minus", -a.fr_variants)):
            tt = np.where(is_fr[ts1[b_all]], thr + d, thr)
            bv = b_all[p[b_all] >= tt]
            Sv, Pv = np.concatenate([ts1[bv], add_s]), np.concatenate([tpj[bv], add_p])
            vd = os.path.join(a.out, "variants", tag)
            os.makedirs(vd, exist_ok=True)
            write_lists(os.path.join(vd, "matching_results.tsv"), t1.entity_id.values, Sv, pd.array(pool_ids, dtype="string"), Pv,
                        "matched_entity_ids")
            nfr = int(is_fr[Sv].sum()); nfr0 = int(is_fr[S].sum())
            log(f"variant {tag}: France threshold {thr + d:.3f} -> France pairs {nfr:,} (main file {nfr0:,}); other countries identical")
    json.dump(dict(holdout_f05=f_ho, thr=thr, gbdt_weight=w, ce_mix=mix, llm_share=v_llm, stacker=stack_model is not None, added=int(len(add_s))), open(os.path.join(a.out, "final_choice.json"), "w"))
    log(f"DONE: {len(S):,} matched pairs ({len(S) / len(t1):.2f} per S1), {n_has.mean():.3f} of S1 with a match; "
        f"by country {pd.Series(n_has).groupby(t1.country.values).mean().round(3).to_dict()}")


if __name__ == "__main__":
    main()
