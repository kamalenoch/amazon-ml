"""Per-S1 set selection maximizing expected F0.5.

F0.5 for one S1 with k predictions, TP hits and T true matches: 1.25*TP / (k + 0.25*T)
(1 if k = T = 0). With calibrated candidate probabilities q (after the one-to-one rule),
E[F] for predicting the top-k is approximated by 1.25*E[TP_k] / (k + 0.25*E[T]),
and the empty set scores P(T = 0) = prod(1 - q). We pick the best k per S1.
"""
import numpy as np
import pandas as pd


def one_to_one(s1_idx, pool_idx, p):
    df = pd.DataFrame({"s1": s1_idx, "pool": pool_idx, "p": p})
    best = df.groupby("pool").p.transform("max")
    return df[df.p >= best].drop_duplicates("pool")


def expected_f_select(s1_idx, pool_idx, p, miss_prior=0.0, min_p=0.0, power=1.0):
    """miss_prior: expected number of true matches not among candidates (blocking misses) per S1."""
    df = one_to_one(s1_idx, pool_idx, p)
    q_all = df.p.values ** power
    df = df.assign(q=q_all).sort_values(["s1", "q"], ascending=[True, False])
    g = df.s1.values
    q = df.q.values
    first = np.r_[True, g[1:] != g[:-1]]
    starts = np.flatnonzero(first)
    grp = np.cumsum(first) - 1
    k = np.arange(len(g)) - starts[grp] + 1
    csum = np.cumsum(q)
    etp = csum - np.r_[0, csum][starts][grp]
    et = np.add.reduceat(q, starts)[grp] + miss_prior
    ef = 1.25 * etp / (k + 0.25 * et)
    logq0 = np.add.reduceat(np.log1p(-np.clip(q, 0, 1 - 1e-7)), starts)
    p_empty = np.exp(logq0) * np.exp(-miss_prior)
    # best k per group
    best_ef = pd.Series(ef).groupby(grp).transform("max").values
    kbest = pd.Series(np.where(ef >= best_ef, k, 10 ** 9)).groupby(grp).transform("min").values
    choose = (k <= kbest) & (best_ef[...] > p_empty[grp]) & (q >= min_p)
    return df[choose][["s1", "pool", "p"]]
