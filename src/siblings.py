"""Stage-2 collective features.

A true S2/S3 record usually has 1-5 sibling records (same S1) that are near-identical to it.
Given stage-1 probabilities p1, compare every candidate with the S1's other top candidates:
if it is a near-duplicate of a confident sibling it is almost surely a match, even when its
own name/number was perturbed. Also adds p1 competition context.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from .features import context_features

TOPS = 3


def _topk_per_group(g, v, k):
    order = np.lexsort((-v, g))
    gs = g[order]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.flatnonzero(first)
    rank = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
    keep = rank < k
    return order[keep], gs[keep], rank[keep]


def sibling_features(c, p1, pool, workers=-1):
    """c: DataFrame [s1_idx, pool_idx] (row-aligned with p1). pool: normalized pool frame."""
    n = len(c)
    s1 = c.s1_idx.values
    pj = c.pool_idx.values
    # top-(TOPS+1) candidates per S1 by p1 (one extra so we can exclude the pair itself)
    rows, gs, rk = _topk_per_group(s1, p1, TOPS + 1)
    top = pd.DataFrame({"s1_idx": gs, "sib_row": rows, "rk": rk})
    pairs = pd.DataFrame({"s1_idx": s1, "row": np.arange(n)}).merge(top, on="s1_idx")
    pairs = pairs[pairs.row != pairs.sib_row]
    pairs = pairs[pairs.groupby("row").cumcount() < TOPS]
    r = pairs.row.values
    sr = pairs.sib_row.values
    a_pool, b_pool = pj[r], pj[sr]
    nm, ad, nu = pool.n_clean.values, pool.a_clean.values, pool.nums.values
    s_name = cpdist(list(nm[a_pool]), list(nm[b_pool]), scorer=fuzz.token_set_ratio, workers=workers, dtype=np.float32)
    s_addr = cpdist(list(ad[a_pool]), list(ad[b_pool]), scorer=fuzz.token_set_ratio, workers=workers, dtype=np.float32)
    s_num = cpdist(list(nu[a_pool]), list(nu[b_pool]), scorer=fuzz.token_set_ratio, workers=workers, dtype=np.float32)
    sp = p1[sr]
    both = (s_name + s_addr) / 2
    df = pd.DataFrame({"row": r, "sn": s_name, "sa": s_addr, "su": s_num, "sb": both, "sp": sp,
                       "wsb": both * sp})
    g = df.groupby("row")
    out = pd.DataFrame(index=np.arange(n))
    agg = g.agg(sib_name_max=("sn", "max"), sib_addr_max=("sa", "max"), sib_num_max=("su", "max"),
                sib_both_max=("sb", "max"), sib_wboth_max=("wsb", "max"), sib_p_max=("sp", "max"),
                sib_p_sum=("sp", "sum"), sib_n=("sp", "size"))
    out = out.join(agg)
    # p1 of the sibling most similar to this record
    idx = df.sb.values.argsort(kind="stable")
    best = df.iloc[idx].drop_duplicates("row", keep="last").set_index("row")
    out["sib_p_of_most_similar"] = best.sp
    out["sib_sim_of_top1"] = df[df.sp >= df.groupby("row").sp.transform("max")].drop_duplicates("row").set_index("row").sb
    out = out.fillna(-1).astype(np.float32)
    ctx = context_features(pd.DataFrame({"s1_idx": s1, "pool_idx": pj, "p1": p1}), "p1", "p1")
    out = pd.concat([out, ctx.reset_index(drop=True)], axis=1)
    out["p1"] = p1.astype(np.float32)
    g1 = pd.Series(p1).groupby(s1)
    out["p1_sum_s1"] = g1.transform("sum").values.astype(np.float32)
    out["p1_n50_s1"] = pd.Series(p1 > 0.5).groupby(s1).transform("sum").values.astype(np.float32)
    return out
