"""Pairwise features for (S1, pool) candidate pairs.

* String similarities: rapidfuzz.process.cpdist (C++, multithreaded, element-wise pairs).
* Set / noise-model / number features: lean Python in forked workers over chunks.
* Competition context (ranks / margins among candidates): vectorized numpy/pandas.
"""
import math
import os
import multiprocessing as mp
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from .noise import token_noise_feats, number_feats, TOKEN_COLS, NUM_COLS

_G = {}


# ------------------------------------------------------------------ IDF tables
def build_idf(series_list):
    c = Counter()
    n = 0
    for s in series_list:
        for v in s:
            n += 1
            c.update(set(v.split()))
    return {k: math.log((n + 1) / (v + 1)) for k, v in c.items()}, math.log(n + 1)


# ------------------------------------------------------------------ vectorized string sims
CPD = [  # (name, column, scorer, joined-without-spaces)
    ("n_ratio", "n_clean", fuzz.ratio, False),
    ("n_tset", "n_clean", fuzz.token_set_ratio, False),
    ("n_tsort", "n_clean", fuzz.token_sort_ratio, False),
    ("n_partial", "n_clean", fuzz.partial_ratio, False),
    ("n_jw_join", "n_clean", JaroWinkler.normalized_similarity, True),
    ("n_ratio_join", "n_clean", fuzz.ratio, True),
    ("n_partial_join", "n_clean", fuzz.partial_ratio, True),
    ("n_skel_tset", "nsk", fuzz.token_set_ratio, False),
    ("a_ratio", "a_clean", fuzz.ratio, False),
    ("a_tset", "a_clean", fuzz.token_set_ratio, False),
    ("a_partial", "a_clean", fuzz.partial_ratio, False),
    ("a_skel_tset", "ask", fuzz.token_set_ratio, False),
    ("u_tset", "nums", fuzz.token_set_ratio, False),
]


def cpdist_features(A, B, workers=-1):
    out = {}
    cache = {}
    for name, col, scorer, join in CPD:
        k = (col, join)
        if k not in cache:
            a, b = A[col], B[col]
            if join:
                cache[k] = ([x.replace(" ", "") for x in a], [x.replace(" ", "") for x in b])
            else:
                cache[k] = (list(a), list(b))
        a, b = cache[k]
        v = cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)
        empty = np.fromiter((not x or not y for x, y in zip(a, b)), bool, len(a))
        v[empty] = -1
        out[name] = v
    return out


# ------------------------------------------------------------------ python per-pair features
def _wjacc(sa, sb, idf, dflt):
    if not sa or not sb:
        return -1.0, -1.0, 0.0
    inter = sa & sb
    wi = sum(idf.get(t, dflt) for t in inter)
    wu = wi + sum(idf.get(t, dflt) for t in sa ^ sb)
    miss = max((idf.get(t, dflt) for t in sa - sb), default=0.0)
    return wi / wu if wu else 0.0, len(inter) / len(sa | sb), miss


def _num_old(n1, n2):
    a, b = n1.split(), n2.split()
    if not a or not b:
        return -1.0, -1.0, -1.0, -1.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    near = 0.0
    if not inter:
        near = float(any(Levenshtein.distance(x, y) <= 1 for x in sa for y in sb if len(x) >= 3 or len(y) >= 3))
    return inter / len(sa | sb), float(a[0] == b[0]), float(inter == 0), near


PY_SRC = ["nsk", "ask", "legal", "nums", "state", "rn", "ra", "ru"]
PY_COLS = (["n_wjacc", "n_jacc", "n_miss_idf", "n_miss_idf_r", "n_first_eq", "n_ntok1", "n_ntok2",
            "legal_eq", "legal_ov", "a_wjacc", "a_jacc", "a_miss_idf", "a_miss_idf_r",
            "num_jacc", "num_first_eq", "num_conflict", "num_near", "state_eq"]
           + ["nn_" + c for c in TOKEN_COLS] + ["an_" + c for c in TOKEN_COLS] + NUM_COLS)


def _py_row(k1, a1, l1, u1, st1, rn1, ra1, ru1, k2, a2, l2, u2, st2, rn2, ra2, ru2):
    idf_n, dn, idf_a, da, nadd, ndrop, aadd, adrop = _G["t"]
    t1, t2 = k1.split(), k2.split()
    s1, s2 = set(t1), set(t2)
    wj, jac, miss = _wjacc(s1, s2, idf_n, dn)
    _, _, miss_r = _wjacc(s2, s1, idf_n, dn)
    x1, x2 = set(a1.split()), set(a2.split())
    if x1 and x2:
        awj, ajac, amiss = _wjacc(x1, x2, idf_a, da)
        _, _, amiss_r = _wjacc(x2, x1, idf_a, da)
    else:
        awj = ajac = amiss = amiss_r = -1.0
    return ((wj, jac, miss, miss_r, float(t1[:1] == t2[:1]), float(len(t1)), float(len(t2)),
             -1.0 if (not l1 or not l2) else float(l1 == l2),
             -1.0 if (not l1 or not l2) else float(bool(set(l1.split()) & set(l2.split()))),
             awj, ajac, amiss, amiss_r)
            + _num_old(u1, u2)
            + (-1.0 if (not st1 or not st2) else float(st1 == st2),)
            + token_noise_feats(rn1.split(), rn2.split(), nadd, ndrop)
            + token_noise_feats(ra1.split(), ra2.split(), aadd, adrop)
            + number_feats(tuple(ru1.split()), tuple(ru2.split())))


def _py_chunk(cols):
    return np.array([_py_row(*r) for r in zip(*cols)], dtype=np.float32)


def _init(tables):
    _G["t"] = tables


def string_features(cands, s1, pool, tables, n_jobs=None, chunk=1_000_000, sub=100_000):
    """tables = (idf_n, dn, idf_a, da, name_add_lo, name_drop_lo, addr_add_lo, addr_drop_lo).
    Returns a float32 DataFrame (one preallocated block, no intermediate copies)."""
    n_jobs = n_jobs or os.cpu_count()
    ii, jj = cands.s1_idx.values, cands.pool_idx.values
    need = sorted({c for _, c, _, _ in CPD} | set(PY_SRC))
    S1 = {c: s1[c].array for c in need}
    PL = {c: pool[c].array for c in need}
    cols = [c for c, *_ in CPD] + PY_COLS
    out = np.empty((len(ii), len(cols)), np.float32)
    ncpd = len(CPD)
    with mp.get_context("fork").Pool(n_jobs, initializer=_init, initargs=(tables,)) as pp:
        for k in range(0, len(ii), chunk):
            i, j = ii[k:k + chunk], jj[k:k + chunk]
            A = {c: np.asarray(S1[c].take(i), dtype=object) for c in need}
            B = {c: np.asarray(PL[c].take(j), dtype=object) for c in need}
            f = cpdist_features(A, B)
            for q, (name, *_) in enumerate(CPD):
                out[k:k + len(i), q] = f[name]
            parts = [[A[c][q:q + sub].tolist() for c in PY_SRC] + [B[c][q:q + sub].tolist() for c in PY_SRC]
                     for q in range(0, len(i), sub)]
            out[k:k + len(i), ncpd:] = np.vstack(pp.map(_py_chunk, parts))
            del A, B, f, parts
    return pd.DataFrame(out, columns=cols, index=cands.index, copy=False)


# ------------------------------------------------------------------ competition context
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


def context_features(c, score_col, prefix):
    """Rank / margin of each pair among its S1's candidates and among the pool record's S1s."""
    from .blocking import _group_rank
    out = pd.DataFrame(index=c.index)
    v = c[score_col].values.astype(np.float32)
    g1, g2 = c.s1_idx.values, c.pool_idx.values
    out[f"{prefix}_rank_s1"] = _group_rank(g1, v).astype(np.float32)
    t1, t2 = _top2(g1, v)
    out[f"{prefix}_gap_s1"] = (t1 - v).astype(np.float32)
    out[f"{prefix}_ratio_s1"] = (v / (t1 + 1e-6)).astype(np.float32)
    out[f"{prefix}_top2gap_s1"] = (t1 - t2).astype(np.float32)
    out[f"{prefix}_rank_pool"] = _group_rank(g2, v).astype(np.float32)
    p1, p2 = _top2(g2, v)
    out[f"{prefix}_gap_pool"] = (p1 - v).astype(np.float32)
    other = np.where(v >= p1, p2, p1)
    out[f"{prefix}_margin_pool"] = (v - other).astype(np.float32)
    return out


def cheap_features(c, s1, pool):
    """Vectorized features that need the FULL candidate set (competition context)."""
    c = c.copy()
    c["bs_total"] = c.bs_name + c.bs_addr + c.bs_cross
    X = pd.DataFrame(index=c.index)
    X["src"] = pool.src.values[c.pool_idx.values].astype(np.float32)
    X["n_cands_s1"] = c.groupby("s1_idx").s1_idx.transform("size").astype(np.float32)
    X["n_cands_pool"] = c.groupby("pool_idx").pool_idx.transform("size").astype(np.float32)
    X["a1_empty"] = (np.asarray(s1.a_clean.str.len().fillna(0).values)[c.s1_idx.values] == 0).astype(np.float32)
    X["a2_empty"] = (np.asarray(pool.a_clean.str.len().fillna(0).values)[c.pool_idx.values] == 0).astype(np.float32)
    for col in ["bs_name", "bs_addr", "bs_cross", "bs_nkeys", "bs_nnk", "bs_nak", "bs_nxk", "bs_total",
                "pr_score", "pr_rank"]:
        if col in c:
            X[col] = c[col].astype(np.float32)
    parts = [X, context_features(c, "bs_total", "bst")]
    if "pr_score" in c:
        parts.append(context_features(c, "pr_score", "prs"))
    return pd.concat(parts, axis=1)
