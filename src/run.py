"""End-to-end pipeline v2.

data -> normalize -> key index (cap 1200 single / 600 composite)
     -> blocking pass on a small S1 slice (wide K) -> train pre-ranker
     -> blocking over ALL S1 with learned pruning (top-K per S1)          [train and test]
     -> noise-model dictionaries learned on S1 group D (labels, out-of-model-sample)
     -> features on S1 group M -> LightGBM (holdout = part of M) -> threshold on holdout macro F0.5
     -> test features in chunks -> predict -> one-to-one selection -> TSVs

Train S1 roles by hash h in [0, 10000):
  h <  PR_END            : pre-ranker training slice (blocking-level labels only)
  h in [M_START, M_END)  : model training (last `holdout` share = holdout)
  h >= D_START           : noise-dictionary learning
Every stage caches to --work; delete a file there to redo it.
"""
import argparse
import gc
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

from .prep import load_split, load_gt
from .blocking import KeyIndex, generate_candidates
from .prerank import prerank_features, make_pruner, PR_COLS
from .features import build_idf, cheap_features, string_features, name_oov
from .noise import LogOddsCounter, EditCounter, frequent_tokens
from .matching import train_lgb, feat_cols, select, f05_macro

T0 = time.time()


def rss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return -1.0


def log(*a):
    print(f"[{time.time() - T0:7.0f}s | {rss_gb():5.1f}GB]", *a, flush=True)


def s1_hash(ids):
    return (pd.util.hash_array(np.asarray(ids, dtype=object)) % 10_000).astype(np.int32)


def labels(c, s1, pool, gtp):
    """Integer-key join (no 77M-row string MultiIndex): pair key = s1_idx * 2^24 + pool_idx."""
    si = pd.Index(s1.entity_id.values).get_indexer(gtp.s1_id.values)
    pj = pd.Index(pool.entity_id.values).get_indexer(gtp.pool_id.values)
    ok = (si >= 0) & (pj >= 0)
    gk = np.unique((si[ok].astype(np.int64) << 24) | pj[ok].astype(np.int64))
    ck = (c.s1_idx.values.astype(np.int64) << 24) | c.pool_idx.values.astype(np.int64)
    return np.isin(ck, gk, assume_unique=False).astype(np.int8)


def write_lists(path, s1_ids, s1_idx, pool_ids_col, pool_idx, col, chunk=200_000):
    """Stream one row per S1 (empty list allowed). Works on integer indices; strings are only
    materialized per chunk of S1 rows. Returns a bool array: S1 row has >= 1 id."""
    s1_ids = np.asarray(s1_ids, dtype=object)
    key = (np.asarray(s1_idx, np.int64) << 24) | np.asarray(pool_idx, np.int64)
    key = np.unique(key)                       # sorted by s1, dedup
    ks, kp = (key >> 24).astype(np.int64), (key & ((1 << 24) - 1)).astype(np.int64)
    bounds = np.searchsorted(ks, np.arange(len(s1_ids) + 1))
    has = np.diff(bounds) > 0
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for r0 in range(0, len(s1_ids), chunk):
            r1 = min(r0 + chunk, len(s1_ids))
            b0, b1 = bounds[r0], bounds[r1]
            pid = np.asarray(pool_ids_col.take(kp[b0:b1]), dtype=object)
            lines = []
            for r in range(r0, r1):
                lo, hi = bounds[r] - b0, bounds[r + 1] - b0
                lines.append(f"{s1_ids[r]}\t{','.join(pid[lo:hi])}\n")
            f.write("".join(lines))
    return has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="output")
    ap.add_argument("--cap", type=int, default=1200, help="pool df cap for single-token keys")
    ap.add_argument("--cap-composite", type=int, default=600, help="pool df cap for composite keys")
    ap.add_argument("--df-scale", type=float, default=1.0)
    ap.add_argument("--wide", default="150,40,40,40", help="k_total,k_name,k_addr,k_cross before pruning")
    ap.add_argument("--k-keep", type=int, default=35, help="candidates kept per S1 after pre-ranking")
    ap.add_argument("--pr-frac", type=float, default=0.05)
    ap.add_argument("--m-frac", type=float, default=0.15, help="share of train S1 used for the model")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--d-frac", type=float, default=0.55, help="share of train S1 used for noise dictionaries")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--predict-chunk", type=int, default=3_000_000)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--fmt", action="store_true", help="add raw-format fingerprint features (FORENSICS-01)")
    ap.add_argument("--extra-keys", action="store_true", help="joined-name suffix + name-pair x number keys")
    ap.add_argument("--dump-xy", action="store_true", help="save the model feature matrix (for learner experiments)")
    ap.add_argument("--numdiff", action="store_true", help="house-number difference structure features (ERR-01)")
    ap.add_argument("--skelnoise", action="store_true", help="noise model on phonetic-skeleton tokens (cross-script)")
    ap.add_argument("--edits", action="store_true", help="one-letter edit class log-odds: typo vs distractor edit")
    ap.add_argument("--k-prior", type=int, default=0, help="reserved wide slots per S1 for empty-address (by name) "
                    "and invented-name (by address) pool records")
    ap.add_argument("--edits-ctx", action="store_true", help="--edits statistics split by lone vs accompanied edit")
    ap.add_argument("--oov", action="store_true", help="pool-name tokens unseen in the country's S1 names (invented names)")
    ap.add_argument("--cos", type=int, default=0, help="cosine-normalized blocking: keep top-N by cosine in the wide "
                                                        "set and feed cosine scores to the pre-ranker (0 = off)")
    ap.add_argument("--d-max-pairs", type=int, default=15_000_000, help="max candidate pairs for noise dictionaries")
    ap.add_argument("--stage", default="all", choices=["all", "train", "test-cands", "predict"],
                    help="train: stop after model.pkl | test-cands: pre-ranker + test blocking only | "
                         "predict: needs model.pkl, prerank.txt, test_cands.parquet in --work")
    ap.add_argument("--extra-work", nargs="*", default=[],
                    help="dirs whose files are copied into --work first (outputs of other jobs)")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    if a.extra_keys:
        import src.blocking as _blk
        _blk.MORE_KEYS = True
    if a.edits_ctx:
        import src.noise as _nz
        _nz.EDIT_CTX = a.edits = True
    import shutil
    for d in a.extra_work:
        for f in os.listdir(d):
            src, dst = os.path.join(d, f), os.path.join(a.work, f)
            if os.path.isfile(src) and not os.path.exists(dst):
                os.symlink(src, dst) if f.endswith(".parquet") else shutil.copy(src, dst)
                log("imported", f, "from", d)
    W = lambda f: os.path.join(a.work, f)
    kt, kn, ka, kx = map(int, a.wide.split(","))
    wide = dict(k_total=kt, k_name=kn, k_addr=ka, k_cross=kx, k_cos=a.cos, k_prior=a.k_prior)

    def with_prior(s1, pool):
        """Pool prior codes for --k-prior (1 empty address, 2 invented name); also caches the --oov columns."""
        if (a.k_prior or a.oov) and "oov_f" not in pool:
            pool["oov_f"], pool["oov_n"] = name_oov(s1, pool)
        if not a.k_prior:
            return None
        empty = np.asarray(pool.a_clean.str.len().fillna(0).values) == 0
        inv = (np.asarray(pool.oov_f.values) >= 1.0) & (np.asarray(pool.oov_n.values) > 0)
        code = np.where(empty, 1, np.where(inv, 2, 0)).astype(np.int8)
        log(f"pool prior: empty address {(code == 1).mean():.3f}, invented name {(code == 2).mean():.3f}")
        return code
    PR_END = int(a.pr_frac * 10_000)
    M_START, M_END = PR_END, PR_END + int(a.m_frac * 10_000)
    D_START = 10_000 - int(a.d_frac * 10_000)
    assert D_START >= M_END, "dictionary S1 group must not overlap the model group"

    def ensure_prerank(s1, pool, gtp, h, ix):
        """Deterministic pre-ranker (same data + params -> same model in every job)."""
        if os.path.exists(W("prerank.txt")):
            return lgb.Booster(model_file=W("prerank.txt"))
        cw = generate_candidates(s1, pool, index=ix, s1_subset=h < PR_END, verbose=False, prior=with_prior(s1, pool), **wide)
        yw = labels(cw, s1, pool, gtp)
        ntrue = gtp.s1_id.isin(set(s1.entity_id.values[h < PR_END])).sum()
        log(f"pre-ranker slice: {len(cw):,} wide pairs, wide recall {yw.sum() / ntrue:.4f}")
        Xw = prerank_features(cw)
        prm = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, verbose=-1,
                   num_threads=4, deterministic=True, force_row_wise=True, seed=7)
        from .blocking import _group_rank
        half = (cw.s1_idx.values % 2) == 0
        m0 = lgb.train(prm, lgb.Dataset(Xw[half], yw[half]), 200)
        r = _group_rank(cw.s1_idx.values[~half], m0.predict(Xw[~half]))
        nt_half = gtp.s1_id.isin(set(s1.entity_id.values[np.unique(cw.s1_idx.values[~half])])).sum()
        for K in (20, 30, 35, 40, 60):
            log(f"  pre-ranker recall@{K} {yw[~half][r < K].sum() / nt_half:.4f} (held-out half of slice)")
        pr = lgb.train(prm, lgb.Dataset(Xw, yw), 200)
        pr.save_model(W("prerank.txt"))
        return pr

    if a.stage == "test-cands" and not os.path.exists(W("prerank.txt")):
        s1, pool = load_split(a.data, "train", a.work, a.jobs)
        _, gtp = load_gt(os.path.join(a.data, "train", "train_ground_truth.tsv"))
        ix = KeyIndex(s1, pool, a.cap, a.df_scale, a.jobs, cap_composite=a.cap_composite)
        ensure_prerank(s1, pool, gtp, s1_hash(s1.entity_id.values), ix)
        del s1, pool, ix
        gc.collect()

    # ================================================================ TRAIN
    if a.stage in ("all", "train") and not os.path.exists(W("model.pkl")):
        s1, pool = load_split(a.data, "train", a.work, a.jobs)
        log("train loaded", len(s1), len(pool))
        _, gtp = load_gt(os.path.join(a.data, "train", "train_ground_truth.tsv"))
        h = s1_hash(s1.entity_id.values)

        if not os.path.exists(W("train_cands.parquet")):
            ix = KeyIndex(s1, pool, a.cap, a.df_scale, a.jobs, cap_composite=a.cap_composite)
            pr = ensure_prerank(s1, pool, gtp, h, ix)
            c = generate_candidates(s1, pool, index=ix, prerank=make_pruner(pr, a.k_keep), prior=with_prior(s1, pool), **wide)
            del ix
            gc.collect()
            c.to_parquet(W("train_cands.parquet"))
        c = pd.read_parquet(W("train_cands.parquet"))
        y_all = labels(c, s1, pool, gtp)
        log(f"train cands {len(c):,}  per S1 {len(c) / len(s1):.1f}  blocking recall {y_all.sum() / len(gtp):.4f}")

        # ---- noise-model dictionaries from group D (not used for model fitting)
        hc = h[c.s1_idx.values]
        d = hc >= D_START
        dsel = np.flatnonzero(d)
        if len(dsel) > a.d_max_pairs:   # plenty for token statistics; bounds time and memory
            dsel = np.sort(np.random.default_rng(0).choice(dsel, a.d_max_pairs, replace=False))
        cnt_n, cnt_a, cnt_s = LogOddsCounter(), LogOddsCounter(), LogOddsCounter()
        cnt_en, cnt_ea = EditCounter(), EditCounter()
        di_all, dj_all, dy_all = c.s1_idx.values[dsel], c.pool_idx.values[dsel], y_all[dsel]
        for k in range(0, len(dsel), 2_000_000):     # stream: never more than 2M pairs of strings alive
            di, dj, dy = di_all[k:k + 2_000_000], dj_all[k:k + 2_000_000], dy_all[k:k + 2_000_000]
            ct = s1.country.array.take(di)
            cnt_n.update(s1.rn.array.take(di), pool.rn.array.take(dj), dy, ct)
            cnt_a.update(s1.ra.array.take(di), pool.ra.array.take(dj), dy, ct)
            if a.skelnoise:
                cnt_s.update(s1.nsk.array.take(di), pool.nsk.array.take(dj), dy, ct)
            if a.edits:
                cnt_en.update(s1.rn.array.take(di), pool.rn.array.take(dj), dy)
                cnt_ea.update(s1.ra.array.take(di), pool.ra.array.take(dj), dy)
        nadd, ndrop = cnt_n.result()
        aadd, adrop = cnt_a.result()
        sadd, sdrop = cnt_s.result() if a.skelnoise else ({}, {})
        edits_t = (cnt_en.result(), cnt_ea.result()) if a.edits else None
        if a.edits:
            show = {k: round(v, 2) for k, v in edits_t[0].items() if k.split("|")[0][-3:] in ("ins", "sub") and k.count("|") == 1}
            log(f"edit classes: name {len(edits_t[0])} addr {len(edits_t[1])}; name (op|position) log-odds {show}")
        del cnt_n, cnt_a, cnt_s, cnt_en, cnt_ea, di_all, dj_all, dy_all, dsel, d
        gc.collect()
        log(f"noise dicts from {min(int((hc >= D_START).sum()), a.d_max_pairs):,} D pairs: name add {len(nadd)} drop {len(ndrop)} | addr add {len(aadd)} drop {len(adrop)}")
        idf_n = build_idf([s1.nsk, pool.nsk])
        idf_a = build_idf([s1.ask, pool.ask])
        fq = (frequent_tokens([s1.rn, pool.rn]), frequent_tokens([s1.ra, pool.ra]))
        tables = (*idf_n, *idf_a, nadd, ndrop, aadd, adrop, *fq)
        extras = tuple(e for e, on in (("numdiff", a.numdiff), ("skel", a.skelnoise), ("edits", a.edits)) if on)
        skel = (sadd, sdrop, frequent_tokens([s1.nsk, pool.nsk])) if a.skelnoise else None

        # ---- features for group M (context computed over ALL candidates, materialized for M only)
        m = (hc >= M_START) & (hc < M_END)
        va_all = hc >= M_START + (1 - a.holdout) * (M_END - M_START)
        # order M rows as [train..., holdout...] so both are contiguous views (no copies)
        idx = np.flatnonzero(m)
        idx = idx[np.argsort(va_all[idx], kind="stable")]
        if a.oov:
            with_prior(s1, pool)
        cheap = cheap_features(c, s1, pool, rows=idx, fmt=a.fmt, oov=a.oov)
        cm, y = c.iloc[idx].reset_index(drop=True), y_all[idx]
        ntr = int((~va_all[idx]).sum())
        del y_all, c, hc, va_all, m
        gc.collect()
        log(f"features on {len(cm):,} model pairs")
        S = string_features(cm, s1, pool, tables, a.jobs, extras=extras, skel=skel, edits=edits_t)
        F = list(S.columns) + list(cheap.columns)
        X = np.empty((len(cm), len(F)), np.float32)
        X[:, :S.shape[1]] = S.values
        X[:, S.shape[1]:] = cheap.values
        del S, cheap
        gc.collect()
        if a.dump_xy:
            np.save(W("xy_X.npy"), X)
            np.save(W("xy_y.npy"), y)
            pd.DataFrame({"s1_idx": cm.s1_idx.values, "pool_idx": cm.pool_idx.values}).to_parquet(W("xy_keys.parquet"))
            json.dump(dict(F=F, ntr=int(ntr)), open(W("xy_meta.json"), "w"))
        log(f"fit lgb: train {ntr:,}  holdout {len(cm) - ntr:,}  pos rate {y.mean():.3f}  feats {len(F)}")
        model = train_lgb(X[:ntr], y[:ntr], rounds=a.rounds, valid=(X[ntr:], y[ntr:]),
                          params=dict(learning_rate=a.lr), feature_name=F)
        va = np.zeros(len(cm), bool)
        va[ntr:] = True
        p = model.predict(X[ntr:], num_threads=0)
        hs1 = np.unique(cm.s1_idx.values[va])
        ids, pids = s1.entity_id.values, pool.entity_id.values
        true = gtp[gtp.s1_id.isin(set(ids[hs1]))].rename(columns={"s1_id": "s1", "pool_id": "pool"})
        pd.DataFrame({"s1_idx": cm.s1_idx.values[va], "pool_idx": cm.pool_idx.values[va], "y": y[va], "p": p}) \
            .to_parquet(W("holdout_preds.parquet"))
        best = (-1, 0.5)
        for thr in np.arange(0.30, 0.96, 0.025):
            sel = select(cm.s1_idx.values[va], cm.pool_idx.values[va], p, thr)
            f, info = f05_macro(pd.DataFrame({"s1": ids[sel.s1.values], "pool": pids[sel.pool.values]}), true, ids[hs1])
            log(f"  thr {thr:.3f}  F0.5 {f:.4f}  {info}")
            best = max(best, (f, thr))
        log(f"HOLDOUT F0.5 {best[0]:.4f} at threshold {best[1]:.3f}")
        imp = pd.Series(model.feature_importance("gain"), F).sort_values(ascending=False)
        log("top features", imp.head(20).round(0).to_dict())
        with open(W("model.pkl"), "wb") as f:
            pickle.dump(dict(model=model, feats=F, thr=float(best[1]), holdout_f05=best[0],
                             noise=(nadd, ndrop, aadd, adrop), fmt=a.fmt, extras=extras,
                             skel_dicts=(sadd, sdrop), edit_tables=edits_t, edit_ctx=a.edits_ctx, oov=a.oov), f)
        json.dump(dict(holdout_f05=best[0], threshold=float(best[1]), best_iter=model.best_iteration,
                       features=F, args=vars(a)), open(W("train_report.json"), "w"), indent=1)
        del X, cm, s1, pool
        gc.collect()

    if a.stage == "train":
        log("stage train done")
        return

    # ================================================================ TEST
    s1, pool = load_split(a.data, "test", a.work, a.jobs)
    log("test loaded", len(s1), len(pool), s1.country.value_counts().to_dict())
    if not os.path.exists(W("test_cands.parquet")):
        pr = lgb.Booster(model_file=W("prerank.txt"))
        ix = KeyIndex(s1, pool, a.cap, a.df_scale, a.jobs, cap_composite=a.cap_composite)
        c = generate_candidates(s1, pool, index=ix, prerank=make_pruner(pr, a.k_keep), prior=with_prior(s1, pool), **wide)
        del ix
        gc.collect()
        c.to_parquet(W("test_cands.parquet"))
    c = pd.read_parquet(W("test_cands.parquet"))
    log(f"test cands {len(c):,}  per S1 {len(c) / len(s1):.1f}")
    if a.stage == "test-cands":
        log("stage test-cands done")
        return
    with open(W("model.pkl"), "rb") as f:
        M = pickle.load(f)
    thr = a.threshold if a.threshold is not None else M["thr"]
    if M.get("edit_ctx"):
        import src.noise as _nz
        _nz.EDIT_CTX = True
    log(f"using threshold {thr:.3f}")
    write_lists(os.path.join(a.out, "candidate_pairs.tsv"), s1.entity_id.values, c.s1_idx.values,
                pool.entity_id.array, c.pool_idx.values, "candidate_entity_ids")
    log("wrote candidate_pairs.tsv")
    if M.get("oov", False) and "oov_f" not in pool:
        pool["oov_f"], pool["oov_n"] = name_oov(s1, pool)
    cheap = cheap_features(c, s1, pool, fmt=M.get("fmt", False), oov=M.get("oov", False))
    idf_n = build_idf([s1.nsk, pool.nsk])
    idf_a = build_idf([s1.ask, pool.ask])
    fq = (frequent_tokens([s1.rn, pool.rn]), frequent_tokens([s1.ra, pool.ra]))
    log(f"frequent tokens (fallback gate): name {len(fq[0]):,} addr {len(fq[1]):,}")
    tables = (*idf_n, *idf_a, *M["noise"], *fq)
    extras = M.get("extras", ())
    skel = (*M["skel_dicts"], frequent_tokens([s1.nsk, pool.nsk])) if "skel" in extras else None
    p = np.zeros(len(c), np.float32)
    ch = cheap.values
    cheap_cols = list(cheap.columns)
    del cheap
    for k in range(0, len(c), a.predict_chunk):
        sl = slice(k, k + a.predict_chunk)
        S = string_features(c.iloc[sl], s1, pool, tables, a.jobs, extras=extras, skel=skel, edits=M.get("edit_tables"))
        X = np.hstack([S.values, ch[sl]])
        assert list(S.columns) + cheap_cols == M["feats"], "feature order mismatch"
        p[sl] = M["model"].predict(X, num_threads=0)
        del S, X
        log(f"  predicted {min(k + a.predict_chunk, len(c)):,}/{len(c):,}")
    pd.DataFrame({"s1_idx": c.s1_idx.values, "pool_idx": c.pool_idx.values, "p": p}).to_parquet(W("test_scores.parquet"))
    sel = select(c.s1_idx.values, c.pool_idx.values, p, thr)
    n = write_lists(os.path.join(a.out, "matching_results.tsv"), s1.entity_id.values, sel.s1.values,
                    pool.entity_id.array, sel.pool.values, "matched_entity_ids")
    log(f"DONE: {len(s1):,} S1 rows, {n.mean():.3f} with >=1 match, {len(sel):,} matched pairs "
        f"({len(sel) / len(s1):.2f} per S1)")
    log("by country (share with match):", pd.Series(n).groupby(np.asarray(s1.country)).mean().round(3).to_dict())


if __name__ == "__main__":
    main()
