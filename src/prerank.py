"""Learned candidate pruning (stage 0): cheap blocking-level features only (S1-side context),
so it can run inside every blocking chunk."""
import numpy as np
import pandas as pd

from .blocking import _group_rank

PR_COLS = ["bs_name", "bs_addr", "bs_cross", "bs_nkeys", "bs_nnk", "bs_nak", "bs_nxk", "bs_total",
           "r_total", "r_name", "r_addr", "r_cross", "q_total", "q_name", "q_addr", "q_cross", "n_s1"]


def prerank_features(b):
    g = b.s1_idx.values
    X = pd.DataFrame({c: b[c].values.astype(np.float32) for c in
                      ["bs_name", "bs_addr", "bs_cross", "bs_nkeys", "bs_nnk", "bs_nak", "bs_nxk"]})
    X["bs_total"] = X.bs_name + X.bs_addr + X.bs_cross
    for k, c in [("total", "bs_total"), ("name", "bs_name"), ("addr", "bs_addr"), ("cross", "bs_cross")]:
        v = X[c].values
        X["r_" + k] = _group_rank(g, v).astype(np.float32)
        mx = pd.Series(v).groupby(g).transform("max").values
        X["q_" + k] = (v / (mx + 1e-6)).astype(np.float32)
    X["n_s1"] = pd.Series(g).groupby(g).transform("size").values.astype(np.float32)
    cols = list(PR_COLS)
    if "bs_cos" in b:                  # cosine-normalized blocking scores (--cos)
        for c in ("bs_cos", "bs_ncos", "bs_acos"):
            v = b[c].values.astype(np.float32)
            X[c] = v
            X["r_" + c] = _group_rank(g, v).astype(np.float32)
            X["q_" + c] = (v / (pd.Series(v).groupby(g).transform("max").values + 1e-6)).astype(np.float32)
            cols += [c, "r_" + c, "q_" + c]
    if "bs_prior" in b:                # reserved-slot records (--k-prior): code + rank within their code
        pc = b.bs_prior.values.astype(np.float32)
        sc = np.where(pc == 1, X.bs_name.values, np.where(pc == 2, X.bs_addr.values, -1.0)).astype(np.float32)
        X["bs_prior"] = pc
        X["r_prior"] = np.where(pc > 0, _group_rank(g, sc), 99).astype(np.float32)
        cols += ["bs_prior", "r_prior"]
    return X[cols]


def make_pruner(model, k_keep, p_keep=None):
    """Keep top-k_keep per S1 by pre-rank score (and anything with score >= p_keep)."""
    def prune(b):
        if len(b) == 0:
            return b
        s = model.predict(prerank_features(b), num_threads=0).astype(np.float32)
        r = _group_rank(b.s1_idx.values, s)
        keep = r < k_keep
        if p_keep is not None:
            keep |= s >= p_keep
        out = b[keep].copy()
        out["pr_score"] = s[keep]
        out["pr_rank"] = r[keep].astype(np.float32)
        return out
    return prune
