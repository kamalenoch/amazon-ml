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

from .normalize import phon, LEGAL as _LEGAL
from .noise import token_noise_feats, number_feats, number_diff_feats, edit_feats, number_sign_feats, TOKEN_COLS, NUM_COLS, NUMD_COLS, EDIT_COLS, NUMS_COLS

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


PY_SRC = ["nsk", "ask", "legal", "nums", "state", "rn", "ra", "ru", "country"]
PY_COLS = (["n_wjacc", "n_jacc", "n_miss_idf", "n_miss_idf_r", "n_first_eq", "n_ntok1", "n_ntok2",
            "legal_eq", "legal_ov", "a_wjacc", "a_jacc", "a_miss_idf", "a_miss_idf_r",
            "num_jacc", "num_first_eq", "num_conflict", "num_near", "state_eq"]
           + ["nn_" + c for c in TOKEN_COLS] + ["an_" + c for c in TOKEN_COLS] + NUM_COLS)


EXTRA_COLS = {"numdiff": NUMD_COLS, "skel": ["ns_" + c for c in TOKEN_COLS],
              "edits": ["ce_" + c for c in EDIT_COLS] + ["cea_" + c for c in EDIT_COLS],
              "numsign": NUMS_COLS, "phon": ["ph_jacc", "ph_cov", "ph_full"]}
EXTRA_ORDER = ("numdiff", "skel", "edits", "numsign", "phon")


def _phon_set(tokens):
    return {phon(t) for t in tokens if len(t) >= 2 and t.isalpha() and t not in _LEGAL} - {""}


def extra_cols(extras):
    return [c for e in EXTRA_ORDER if e in extras for c in EXTRA_COLS[e]]


def _py_row(k1, a1, l1, u1, st1, rn1, ra1, ru1, ct1, k2, a2, l2, u2, st2, rn2, ra2, ru2, ct2):
    idf_n, dn, idf_a, da, nadd, ndrop, aadd, adrop, fq_n, fq_a = _G["t"]
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
    row = ((wj, jac, miss, miss_r, float(t1[:1] == t2[:1]), float(len(t1)), float(len(t2)),
             -1.0 if (not l1 or not l2) else float(l1 == l2),
             -1.0 if (not l1 or not l2) else float(bool(set(l1.split()) & set(l2.split()))),
             awj, ajac, amiss, amiss_r)
            + _num_old(u1, u2)
            + (-1.0 if (not st1 or not st2) else float(st1 == st2),)
            + token_noise_feats(rn1.split(), rn2.split(), nadd, ndrop, ct1.lower(), fq_n)
            + token_noise_feats(ra1.split(), ra2.split(), aadd, adrop, ct1.lower(), fq_a)
            + number_feats(tuple(ru1.split()), tuple(ru2.split())))
    ex = _G.get("extras", ())
    if "numdiff" in ex:
        row += number_diff_feats(u1, u2)
    if "skel" in ex:          # script-independent noise model: add/drop on phonetic-skeleton tokens
        sadd, sdrop, fq_s = _G["skel"]
        row += token_noise_feats(t1, t2, sadd, sdrop, ct1.lower(), fq_s)
    if "edits" in ex:         # one-letter edit class log-odds: typo (true copy) vs distractor edit
        en, ea = _G["edits"]
        row += edit_feats(rn1.split(), rn2.split(), en) + edit_feats(ra1.split(), ra2.split(), ea)
    if "numsign" in ex:       # signed house-number delta + learned log-odds of the exact delta
        row += number_sign_feats(u1, u2, ct1.lower(), _G["numsign"])
    if "phon" in ex:          # name overlap on sound keys: English vs native-script romanization
        p1, p2 = _phon_set(rn1.split()), _phon_set(rn2.split())
        if p1 and p2:
            inter = len(p1 & p2)
            row += (inter / len(p1 | p2), inter / len(p1), float(p1 == p2))
        else:
            row += (-1.0, -1.0, -1.0)
    return row


def _py_chunk(cols):
    return np.array([_py_row(*r) for r in zip(*cols)], dtype=np.float32)


def _init(tables, extras=(), skel=None, edits=None, numsign=None):
    _G["t"] = tables
    _G["extras"] = tuple(extras)
    _G["skel"] = skel
    _G["edits"] = edits
    _G["numsign"] = numsign


def string_feature_cols(extras=()):
    return [c for c, *_ in CPD] + PY_COLS + extra_cols(extras)


def string_features(cands, s1, pool, tables, n_jobs=None, chunk=1_000_000, sub=100_000, extras=(), skel=None,
                    edits=None, numsign=None, out=None):
    """tables = (idf_n, dn, idf_a, da, name_add_lo, name_drop_lo, addr_add_lo, addr_drop_lo,
                  frequent_name_tokens, frequent_addr_tokens).
    Returns a float32 DataFrame (one preallocated block, no intermediate copies).
    out: optional (n, len(string_feature_cols(extras))) float32 view to fill in place, e.g. the left block of
    the training matrix, so the string features never exist twice in memory."""
    n_jobs = n_jobs or os.cpu_count()
    ii, jj = cands.s1_idx.values, cands.pool_idx.values
    need = sorted({c for _, c, _, _ in CPD} | set(PY_SRC))
    S1 = {c: s1[c].array for c in need}
    PL = {c: pool[c].array for c in need}
    cols = string_feature_cols(extras)
    if out is None:
        out = np.empty((len(ii), len(cols)), np.float32)
    elif out.shape != (len(ii), len(cols)):
        raise ValueError(f"out has shape {out.shape}, expected {(len(ii), len(cols))}")
    ncpd = len(CPD)
    with mp.get_context("fork").Pool(n_jobs, initializer=_init, initargs=(tables, extras, skel, edits, numsign)) as pp:
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


CTX = ["rank_s1", "gap_s1", "ratio_s1", "top2gap_s1", "rank_pool", "gap_pool", "margin_pool"]


def _context_into(out, col0, g1, g2, v, rows):
    """Write the 7 competition-context features of score v into out[:, col0:col0+7] (rows subset)."""
    from .blocking import _group_rank
    sel = (lambda x: x) if rows is None else (lambda x: x[rows])
    out[:, col0 + 0] = sel(_group_rank(g1, v))
    t1, t2 = _top2(g1, v)
    vv = sel(v)
    out[:, col0 + 1] = sel(t1) - vv
    out[:, col0 + 2] = vv / (sel(t1) + 1e-6)
    out[:, col0 + 3] = sel(t1) - sel(t2)
    del t1, t2
    out[:, col0 + 4] = sel(_group_rank(g2, v))
    p1, p2 = _top2(g2, v)
    p1s, p2s = sel(p1), sel(p2)
    del p1, p2
    out[:, col0 + 5] = p1s - vv
    out[:, col0 + 6] = vv - np.where(vv >= p1s, p2s, p1s)


def context_features(c, score_col, prefix):
    v = c[score_col].values.astype(np.float32)
    out = np.empty((len(c), len(CTX)), np.float32)
    _context_into(out, 0, c.s1_idx.values, c.pool_idx.values, v, None)
    return pd.DataFrame(out, columns=[f"{prefix}_{k}" for k in CTX], index=c.index)


BASE_CHEAP = ["bs_name", "bs_addr", "bs_cross", "bs_nkeys", "bs_nnk", "bs_nak", "bs_nxk", "pr_score", "pr_rank",
              "bs_cos", "bs_ncos", "bs_acos"]


PF_BITS = ["allcaps", "lower", "leadjunk", "bracket", "dblspace", "domain", "nonascii", "digit", "alias", "idtag"]
PA_BITS = ["empty", "allcaps", "masked", "nulltok", "dblspace", "zeropad", "landmark"]
FMT_FEATS = ["pf_" + b for b in PF_BITS] + ["pa_" + b for b in PA_BITS] + ["fmt_noise", "raw_name_eq", "raw_addr_eq"]


def fmt_features(c, s1, pool, rows=None, out=None):
    """Raw-format fingerprints of the pool record + raw-string equality with the S1 record.
    out: optional (n, len(FMT_FEATS)) float32 view to fill in place (no second full-size block)."""
    si, pj = c.s1_idx.values, c.pool_idx.values
    if rows is not None:
        si, pj = si[rows], pj[rows]
    fn = np.asarray(pool.fn.values)[pj].astype(np.int32)
    fa = np.asarray(pool.fa.values)[pj].astype(np.int32)
    if out is None:
        out = np.empty((len(si), len(FMT_FEATS)), np.float32)
    for b in range(len(PF_BITS)):
        out[:, b] = (fn >> b) & 1
    for b in range(len(PA_BITS)):
        out[:, len(PF_BITS) + b] = (fa >> b) & 1
    k = len(PF_BITS) + len(PA_BITS)
    noisy_n = sum(((fn >> b) & 1) for b in (0, 1, 2, 3, 4, 5, 6, 9))
    noisy_a = sum(((fa >> b) & 1) for b in (1, 2, 3, 4, 5))
    out[:, k] = noisy_n + noisy_a
    out[:, k + 1] = np.asarray(s1.hn.values)[si] == np.asarray(pool.hn.values)[pj]
    out[:, k + 2] = np.asarray(s1.ha.values)[si] == np.asarray(pool.ha.values)[pj]
    return out


OOV_FEATS = ["pn_oov_frac", "pn_oov_n", "pn_oov_all"]


def name_oov(s1, pool):
    """Per pool record: share and count of its clean-name tokens that no S1 name of the same country uses.
    The generator replaces some true copies' names with invented words ('Jaxhalo', 'Belodelta'); 97-98% of
    pool records named by one unseen word are true copies (ERR-02), distractors reuse real vocabulary."""
    frac = np.zeros(len(pool), np.float32)
    cnt = np.zeros(len(pool), np.float32)
    pc = np.asarray(pool.country.values)
    sc = np.asarray(s1.country.values)
    for ct in pd.unique(pc):
        voc = {t for n in pd.unique(np.asarray(s1.n_clean.values)[sc == ct]) if isinstance(n, str) for t in n.split()}
        rows = np.flatnonzero(pc == ct)
        names = np.asarray(pool.n_clean.values)[rows]
        memo = {}
        for r, n in zip(rows, names):
            v = memo.get(n)
            if v is None:
                t = n.split() if isinstance(n, str) else []
                k = sum(x not in voc for x in t)
                v = memo[n] = (k / len(t) if t else 0.0, float(k))
            frac[r], cnt[r] = v
    return frac, cnt


def cheap_features(c, s1, pool, rows=None, fmt=False, oov=False):
    """Features that need the FULL candidate set (competition context), materialized only for `rows`
    (bool mask or index array; None = all). One preallocated float32 block, no DataFrame copies:
    at 77M candidate pairs this is the difference between ~5 GB and an OOM kill."""
    if rows is not None and np.asarray(rows).dtype == bool:
        rows = np.flatnonzero(rows)
    n = len(c) if rows is None else len(rows)
    sel = (lambda x: x) if rows is None else (lambda x: x[rows])
    base = [b for b in BASE_CHEAP if b in c]
    has_pr = "pr_score" in c
    cols = (["src", "n_cands_s1", "n_cands_pool", "a1_empty", "a2_empty"] + base + ["bs_total"]
            + [f"bst_{k}" for k in CTX] + ([f"prs_{k}" for k in CTX] if has_pr else [])
            + (FMT_FEATS if fmt else []) + (OOV_FEATS if oov else []))
    out = np.empty((n, len(cols)), np.float32)
    g1, g2 = c.s1_idx.values, c.pool_idx.values
    si, pj = sel(g1), sel(g2)
    out[:, 0] = pool.src.values[pj]
    out[:, 1] = np.bincount(g1, minlength=len(s1))[si]
    out[:, 2] = np.bincount(g2, minlength=len(pool))[pj]
    out[:, 3] = (np.asarray(s1.a_clean.str.len().fillna(0).values) == 0)[si]
    out[:, 4] = (np.asarray(pool.a_clean.str.len().fillna(0).values) == 0)[pj]
    k = 5
    for b in base:
        out[:, k] = sel(c[b].values)
        k += 1
    tot = (c.bs_name.values + c.bs_addr.values + c.bs_cross.values).astype(np.float32)
    out[:, k] = sel(tot)
    k += 1
    _context_into(out, k, g1, g2, tot, rows)
    del tot
    k += len(CTX)
    if has_pr:
        _context_into(out, k, g1, g2, c.pr_score.values.astype(np.float32), rows)
        k += len(CTX)
    if fmt:
        fmt_features(c, s1, pool, rows, out=out[:, k:k + len(FMT_FEATS)])
        k += len(FMT_FEATS)
    if oov:
        of, on = np.asarray(pool.oov_f.values)[pj], np.asarray(pool.oov_n.values)[pj]
        out[:, k], out[:, k + 1], out[:, k + 2] = of, on, (of >= 1.0) & (on > 0)
    return pd.DataFrame(out, columns=cols, copy=False)
