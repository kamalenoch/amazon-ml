"""Pairwise features for (S1, pool) candidate pairs.

String features are computed with rapidfuzz in forked worker processes over chunks of
pairs (records are shared copy-on-write via module globals). Context features (ranks /
competition among candidates) are vectorized with pandas groupby.
"""
import math
import os
import multiprocessing as mp
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .normalize import skeleton

_G = {}  # per-worker state (idf tables)


def build_idf(series_list):
    c = Counter()
    n = 0
    for s in series_list:
        for v in s:
            n += 1
            c.update(set(skeleton(t) for t in v.split()))
    return {k: math.log((n + 1) / (v + 1)) for k, v in c.items()}, math.log(n + 1)


def _skels(s):
    return [skeleton(t) for t in s.split()] if s else []


def _wjacc(a, b, idf, dflt):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return -1.0, -1.0, 0.0
    inter = sa & sb
    wi = sum(idf.get(t, dflt) for t in inter)
    wu = sum(idf.get(t, dflt) for t in sa | sb)
    # max idf of an S1 token that is missing in the pool record (rare token unexplained)
    miss = max((idf.get(t, dflt) for t in sa - sb), default=0.0)
    return wi / wu if wu else 0.0, len(inter) / len(sa | sb), miss


def _num_feats(n1, n2):
    a, b = n1.split(), n2.split()
    if not a or not b:
        return -1.0, -1.0, -1.0, -1.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    first_eq = float(a[0] == b[0])
    near = 0.0
    if not inter:  # typo'd house number: 4242 vs 424, 6176 vs 1176
        near = float(any(Levenshtein.distance(x, y) <= 1 for x in sa for y in sb if len(x) >= 3 or len(y) >= 3))
    return inter / len(sa | sb), first_eq, float(inter == 0), near


def _pair_feats(n1, a1, u1, st1, l1, n2, a2, u2, st2, l2):
    idf_n, dn, idf_a, da = _G["idf_n"], _G["dn"], _G["idf_a"], _G["da"]
    j1, j2 = n1.replace(" ", ""), n2.replace(" ", "")
    k1, k2 = _skels(n1), _skels(n2)
    wj, jac, miss = _wjacc(k1, k2, idf_n, dn)
    wj_r, _, miss_r = _wjacc(k2, k1, idf_n, dn)
    legal_eq = -1.0 if (not l1 or not l2) else float(l1 == l2)
    legal_ov = -1.0 if (not l1 or not l2) else float(bool(set(l1.split()) & set(l2.split())))
    if a1 and a2:
        ar = fuzz.ratio(a1, a2)
        ats = fuzz.token_set_ratio(a1, a2)
        apr = fuzz.partial_ratio(a1, a2) if len(a1) < 200 and len(a2) < 200 else ar
        awj, ajac, amiss = _wjacc(_skels(a1), _skels(a2), idf_a, da)
        _, _, amiss_r = _wjacc(_skels(a2), _skels(a1), idf_a, da)
    else:
        ar = ats = apr = awj = ajac = amiss = amiss_r = -1.0
    nj, nfe, nconf, nnear = _num_feats(u1, u2)
    st = -1.0 if (not st1 or not st2) else float(st1 == st2)
    return (
        fuzz.ratio(n1, n2), fuzz.token_set_ratio(n1, n2), fuzz.token_sort_ratio(n1, n2),
        fuzz.partial_ratio(n1, n2), JaroWinkler.similarity(j1, j2) if j1 and j2 else 0.0,
        fuzz.ratio(j1, j2), fuzz.partial_ratio(j1, j2) if j1 and j2 else 0.0,
        fuzz.token_set_ratio(" ".join(k1), " ".join(k2)),
        wj, jac, miss, miss_r, float(k1[:1] == k2[:1]),
        len(k1), len(k2), legal_eq, legal_ov,
        ar, ats, apr, awj, ajac, amiss, amiss_r,
        nj, nfe, nconf, nnear, st,
    )


STR_COLS = ["n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw_join", "n_ratio_join", "n_partial_join",
            "n_skel_tset", "n_wjacc", "n_jacc", "n_miss_idf", "n_miss_idf_r", "n_first_eq", "n_ntok1",
            "n_ntok2", "legal_eq", "legal_ov", "a_ratio", "a_tset", "a_partial", "a_wjacc", "a_jacc",
            "a_miss_idf", "a_miss_idf_r", "num_jacc", "num_first_eq", "num_conflict", "num_near", "state_eq"]


STR_SRC = ["n_clean", "a_clean", "nums", "state", "legal"]


def _chunk(cols):
    return np.array([_pair_feats(*r) for r in zip(*cols)], dtype=np.float32)


def _init(idf_n, idf_a):
    _G["idf_n"], _G["dn"] = idf_n
    _G["idf_a"], _G["da"] = idf_a


def string_features(cands, s1, pool, idf_n, idf_a, n_jobs=None, chunk=100_000):
    """Parent slices the needed strings per chunk and ships them to workers, so worker
    memory stays bounded regardless of dataset size."""
    n_jobs = n_jobs or os.cpu_count()
    A = {c: s1[c].values for c in STR_SRC}
    B = {c: pool[c].values for c in STR_SRC}
    ii, jj = cands.s1_idx.values, cands.pool_idx.values

    def parts():
        for k in range(0, len(ii), chunk):
            i, j = ii[k:k + chunk], jj[k:k + chunk]
            yield [A[c][i].tolist() for c in STR_SRC] + [B[c][j].tolist() for c in STR_SRC]

    if n_jobs > 1 and len(ii) > chunk:
        with mp.get_context("fork").Pool(n_jobs, initializer=_init, initargs=(idf_n, idf_a)) as p:
            res = list(p.imap(_chunk, parts(), chunksize=1))
    else:
        _init(idf_n, idf_a)
        res = [_chunk(x) for x in parts()]
    X = np.vstack(res) if res else np.zeros((0, len(STR_COLS)), np.float32)
    return pd.DataFrame(X, columns=STR_COLS, index=cands.index)


def context_features(c, score_col, prefix):
    """Rank / margin of each pair among its S1's candidates and among the pool record's S1s."""
    out = pd.DataFrame(index=c.index)
    g1 = c.groupby("s1_idx")[score_col]
    out[f"{prefix}_rank_s1"] = g1.rank(ascending=False, method="first").astype(np.float32)
    mx = g1.transform("max")
    out[f"{prefix}_gap_s1"] = (mx - c[score_col]).astype(np.float32)
    out[f"{prefix}_ratio_s1"] = (c[score_col] / (mx + 1e-6)).astype(np.float32)
    g2 = c.groupby("pool_idx")[score_col]
    out[f"{prefix}_rank_pool"] = g2.rank(ascending=False, method="first").astype(np.float32)
    mx2 = g2.transform("max")
    out[f"{prefix}_gap_pool"] = (mx2 - c[score_col]).astype(np.float32)
    # margin over the best *other* S1 competing for this pool record
    top1, top2 = _top2(c.pool_idx.values, c[score_col].values)
    v = c[score_col].values
    other = np.where(v >= top1, top2, top1)
    out[f"{prefix}_margin_pool"] = (v - other).astype(np.float32)
    t1, t2 = _top2(c.s1_idx.values, v)
    out[f"{prefix}_top2gap_s1"] = (t1 - t2).astype(np.float32)
    return out


def _top2(g, v):
    """Per-row: best and second-best value within the row's group (second=-1 if none)."""
    order = np.lexsort((-v, g))
    gs, vs = g[order], v[order]
    first = np.r_[True, gs[1:] != gs[:-1]]
    grp = np.cumsum(first) - 1
    starts = np.flatnonzero(first)
    sizes = np.diff(np.r_[starts, len(g)])
    b1 = vs[starts]
    b2 = np.where(sizes > 1, vs[np.minimum(starts + 1, len(g) - 1)], -1.0)
    t1, t2 = np.empty(len(g), vs.dtype), np.empty(len(g), vs.dtype)
    t1[order], t2[order] = b1[grp], b2[grp]
    return t1, t2


def cheap_features(c, s1, pool):
    """Vectorized features that need the FULL candidate set (competition context)."""
    c = c.copy()
    c["bs_total"] = c.bs_name + c.bs_addr + c.bs_cross
    X = pd.DataFrame(index=c.index)
    X["src"] = pool.src.values[c.pool_idx.values].astype(np.float32)
    X["n_cands_s1"] = c.groupby("s1_idx").s1_idx.transform("size").astype(np.float32)
    X["n_cands_pool"] = c.groupby("pool_idx").pool_idx.transform("size").astype(np.float32)
    X["a1_empty"] = (s1.a_clean.values[c.s1_idx.values] == "").astype(np.float32)
    X["a2_empty"] = (pool.a_clean.values[c.pool_idx.values] == "").astype(np.float32)
    for col in ["bs_name", "bs_addr", "bs_cross", "bs_nkeys", "bs_total"]:
        X[col] = c[col].astype(np.float32)
    return pd.concat([X, context_features(c, "bs_total", "bst")], axis=1)


def all_features(cands, s1, pool, idf_n, idf_a, n_jobs=None):
    X = cheap_features(cands, s1, pool)
    S = string_features(cands, s1, pool, idf_n, idf_a, n_jobs)
    return pd.concat([S, X], axis=1)
