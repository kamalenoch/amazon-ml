"""Learned candidate pruning (stage 0).

v5: besides blocking-level statistics the pre-ranker sees four cheap string similarities
(rapidfuzz cpdist, C++), so it can rank candidates that tie on blocking score - e.g. an empty-address
pool record whose candidates all share the same name key. It runs inside every blocking chunk; its
score also feeds the pool-side top-k tracker and the prs_* competition-context features of the model."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from .blocking import _group_rank

BS_COLS = ["bs_name", "bs_addr", "bs_cross", "bs_nkeys", "bs_nnk", "bs_nak", "bs_nxk", "bs_total",
           "r_total", "r_name", "r_addr", "r_cross", "q_total", "q_name", "q_addr", "q_cross", "n_s1"]
STR_SPEC = [  # (name, column, scorer, joined-without-spaces)
    ("ps_n_tset", "n_clean", fuzz.token_set_ratio, False),
    ("ps_n_join", "n_clean", fuzz.ratio, True),
    ("ps_a_tset", "a_clean", fuzz.token_set_ratio, False),
    ("ps_u_tset", "nums", fuzz.token_set_ratio, False),
]
STR_COLS = [n for n, *_ in STR_SPEC] + ["ps_sum", "r_ps", "q_ps", "gap_ps"]
PR_COLS = BS_COLS + STR_COLS


class StringSims:
    """cpdist similarities for (s1_idx, pool_idx) rows; -1 where either side is empty."""

    def __init__(self, s1, pool, sub=1_000_000):
        self.sub = sub
        cols = sorted({c for _, c, _, _ in STR_SPEC})
        self.A = {c: s1[c].array for c in cols}
        self.B = {c: pool[c].array for c in cols}
        self.eA = {c: np.asarray(s1[c].str.len().fillna(0).values) == 0 for c in cols}
        self.eB = {c: np.asarray(pool[c].str.len().fillna(0).values) == 0 for c in cols}

    def __call__(self, ii, jj):
        out = np.empty((len(ii), len(STR_SPEC)), np.float32)
        for k in range(0, len(ii), self.sub):
            i, j = ii[k:k + self.sub], jj[k:k + self.sub]
            cache = {}
            for q, (name, col, scorer, join) in enumerate(STR_SPEC):
                if col not in cache:
                    cache[col] = (self.A[col].take(i).to_numpy(), self.B[col].take(j).to_numpy())
                a, b = cache[col]
                if join:
                    a, b = [x.replace(" ", "") for x in a], [x.replace(" ", "") for x in b]
                v = cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
                v[self.eA[col][i] | self.eB[col][j]] = -1
                out[k:k + len(i), q] = v
            del cache
        return out


def prerank_features(b, sims):
    """b: candidate rows with bs_* columns. Returns a float32 matrix [len(b), len(PR_COLS)] (preallocated,
    so peak memory stays ~1x the output)."""
    g = b.s1_idx.values
    X = np.empty((len(b), len(PR_COLS)), np.float32)
    col = {c: q for q, c in enumerate(PR_COLS)}
    for c in ["bs_name", "bs_addr", "bs_cross", "bs_nkeys", "bs_nnk", "bs_nak", "bs_nxk"]:
        X[:, col[c]] = b[c].values
    X[:, col["bs_total"]] = X[:, col["bs_name"]] + X[:, col["bs_addr"]] + X[:, col["bs_cross"]]
    # group (per-S1) statistics via sorted segments
    order = np.argsort(g, kind="stable")
    gs = g[order]
    first = np.r_[True, gs[1:] != gs[:-1]]
    starts = np.flatnonzero(first)
    seg = np.cumsum(first) - 1

    def gmax(v):
        m = np.empty(len(v), np.float32)
        m[order] = np.maximum.reduceat(v[order], starts)[seg]
        return m

    for k, c in [("total", "bs_total"), ("name", "bs_name"), ("addr", "bs_addr"), ("cross", "bs_cross")]:
        v = X[:, col[c]]
        X[:, col["r_" + k]] = _group_rank(g, v)
        X[:, col["q_" + k]] = v / (gmax(v) + 1e-6)
    sizes = np.diff(np.r_[starts, len(g)])
    n = np.empty(len(g), np.float32)
    n[order] = sizes[seg]
    X[:, col["n_s1"]] = n
    S = sims(b.s1_idx.values, b.pool_idx.values)
    for q, (name, *_) in enumerate(STR_SPEC):
        X[:, col[name]] = S[:, q]
    v = (np.maximum(S[:, 0], 0) + np.maximum(S[:, 1], 0) + np.maximum(S[:, 2], 0) + 0.5 * np.maximum(S[:, 3], 0))
    del S
    mx = gmax(v)
    X[:, col["ps_sum"]] = v
    X[:, col["r_ps"]] = _group_rank(g, v)
    X[:, col["q_ps"]] = v / (mx + 1e-6)
    X[:, col["gap_ps"]] = mx - v
    return X


def _s1_blocks(g, step):
    """Split row range into pieces of ~step rows that never cut an S1 group (g sorted)."""
    cuts = [0]
    while cuts[-1] < len(g):
        nxt = min(cuts[-1] + step, len(g))
        if nxt < len(g):
            nxt = int(np.searchsorted(g, g[nxt], side="left"))
            if nxt <= cuts[-1]:
                nxt = int(np.searchsorted(g, g[cuts[-1]], side="right"))
        cuts.append(nxt)
    return list(zip(cuts[:-1], cuts[1:]))


def prerank_scores(model, b, sims, step=2_000_000):
    """Pre-ranker scores for all rows of b, computed in S1-aligned pieces (bounded memory)."""
    g = b.s1_idx.values
    if len(g) > 1 and (np.diff(g) < 0).any():
        order = np.argsort(g, kind="stable")
        s = np.empty(len(g), np.float32)
        s[order] = prerank_scores(model, b.iloc[order], sims, step)
        return s
    out = np.empty(len(g), np.float32)
    for a, z in _s1_blocks(g, step):
        out[a:z] = model.predict(prerank_features(b.iloc[a:z], sims), num_threads=0)
    return out


def make_pruner(model, k_keep, sims, p_keep=None):
    """Scores every wide pair of a chunk; generate_candidates keeps the S1-side top-k_keep and feeds all
    scored rows to the global pool-side top-k tracker."""
    def score(b):
        if len(b) == 0:
            return b.assign(pr_score=np.zeros(0, np.float32), pr_rank=np.zeros(0, np.float32))
        s = prerank_scores(model, b, sims)
        r = _group_rank(b.s1_idx.values, s)
        return b.assign(pr_score=s, pr_rank=r.astype(np.float32))
    score.k_keep = k_keep
    score.p_keep = p_keep
    return score
