"""Candidate generation: IDF-weighted multi-key token blocking, fully vectorized.

Every record emits a set of hashed keys (country-scoped):
  name  : phonetic-skeleton tokens, 5-char prefixes, skeleton token pairs
  addr  : skeleton tokens, (house-number, token) pairs
An S1 record and a pool record share a key -> they accumulate idf(key) into a name or
address score. Keys that are too common (pool df > cap) are dropped. For every S1 we
keep the union of the top-K by total / name / address score. All joins are numpy
range-expansions over sorted key arrays (no Python loop per pair).
"""
import os
import multiprocessing as mp
import numpy as np
import pandas as pd

from .normalize import skeleton

STREET_TYPES = {"st", "ave", "rd", "dr", "ln", "cir", "ct", "blvd", "pl", "pkwy", "hwy", "ter", "trl",
                "sq", "cv", "way", "loop", "n", "s", "e", "w", "ne", "nw", "se", "sw", "ste", "apt", "flr",
                "bldg", "sec", "ngr", "col", "marg", "rue", "all", "chem", "imp", "rte", "fbg", "quai",
                "crs", "res", "main", "new", "old"}
MAX_PAIR_TOKS = 6


MAX_CROSS_NAME, MAX_CROSS_ADDR, MAX_ADDR_PAIR = 4, 6, 6


def record_keys(country, n_clean, a_clean, nums):
    """Returns (name_keys, addr_keys, cross_keys) as sets of strings."""
    c = country
    nk, ak, xk = set(), set(), set()
    ntoks = n_clean.split()
    sk = []
    for t in ntoks:
        s = skeleton(t)
        if len(s) >= 2:
            nk.add(f"t{c}|{s}")
            sk.append(s)
        elif len(t) >= 2:
            nk.add(f"r{c}|{t}")
        if len(t) >= 5:
            nk.add(f"p{c}|{t[:5]}")
    joined = "".join(ntoks)
    if len(joined) >= 8:                       # 'southernexports.com' <-> 'Southern Exports'
        nk.add(f"j{c}|{joined[:8]}")
    sk = list(dict.fromkeys(sk))[:MAX_PAIR_TOKS]
    for i in range(len(sk)):
        for j in range(i + 1, len(sk)):
            a, b = (sk[i], sk[j]) if sk[i] < sk[j] else (sk[j], sk[i])
            nk.add(f"b{c}|{a}|{b}")
    if a_clean or nums:
        ask = []
        for t in a_clean.split():
            if t in STREET_TYPES:
                continue
            s = skeleton(t) if t.isalpha() else t
            if len(s) >= 2:
                ask.append(s)
        ask = list(dict.fromkeys(ask))
        for s in ask:
            ak.add(f"a{c}|{s}")
        nl = nums.split()[:2]
        for n in nl:
            for s in ask:
                ak.add(f"n{c}|{n}|{s}")
        ap = ask[:MAX_ADDR_PAIR]
        for i in range(len(ap)):
            for j in range(i + 1, len(ap)):
                a, b = (ap[i], ap[j]) if ap[i] < ap[j] else (ap[j], ap[i])
                ak.add(f"q{c}|{a}|{b}")
        for s_ in sk[:MAX_CROSS_NAME]:
            for a in ask[:MAX_CROSS_ADDR] + nl:
                xk.add(f"x{c}|{s_}|{a}")
    return nk, ak, xk


def _keys_chunk(args):
    off, countries, names, addrs, nums = args
    recs, keys, fams = [], [], []
    for i, (c, n, a, u) in enumerate(zip(countries, names, addrs, nums)):
        for fam, ks in enumerate(record_keys(c.lower(), n, a, u)):
            for k in ks:
                recs.append(off + i); keys.append(k); fams.append(fam)
    h = pd.util.hash_array(np.array(keys, dtype=object), categorize=False)
    return np.array(recs, np.int32), h, np.array(fams, np.int8)


def build_keys(df, n_jobs=None, chunk=100_000):
    n_jobs = n_jobs or os.cpu_count()
    cols = [df.country.tolist(), df.n_clean.tolist(), df.a_clean.tolist(), df.nums.tolist()]
    parts = [(i, *[c[i:i + chunk] for c in cols]) for i in range(0, len(df), chunk)]
    if n_jobs > 1 and len(parts) > 1:
        with mp.get_context("fork").Pool(n_jobs) as pool:
            res = pool.map(_keys_chunk, parts)
    else:
        res = [_keys_chunk(p) for p in parts]
    return (np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]),
            np.concatenate([r[2] for r in res]))


def _group_rank(g, score):
    """rank (0 = best) of each row within its group g, by descending score."""
    order = np.lexsort((-score, g))
    gs = g[order]
    first = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    rank_sorted = np.arange(len(g)) - np.repeat(first, np.diff(np.r_[first, len(g)]))
    rank = np.empty(len(g), np.int32)
    rank[order] = rank_sorted
    return rank


def generate_candidates(s1, pool, cap=600, df_scale=1.0, k_total=30, k_name=10, k_addr=10,
                        budget=60_000_000, n_jobs=None, verbose=True):
    """Returns DataFrame [s1_idx, pool_idx, bs_name, bs_addr, bs_nkeys] (row indices into s1/pool)."""
    import time
    t0 = time.time()
    pr, pk, pf = build_keys(pool, n_jobs)
    sr, sk, sf = build_keys(s1, n_jobs)
    if verbose:
        print(f"[block] keys pool={len(pk):,} s1={len(sk):,} ({time.time() - t0:.0f}s)", flush=True)
    order = np.argsort(pk)
    pk = pk[order]
    pr = pr[order]
    del order, pf
    newk = np.empty(len(pk), bool)
    newk[0] = True
    np.not_equal(pk[1:], pk[:-1], out=newk[1:])
    starts = np.flatnonzero(newk)
    uk = pk[starts]
    del pk, newk
    counts = np.diff(np.r_[starts, len(pr)]).astype(np.int64)
    idf = np.log((len(pool) + 1) / (counts + 1)).astype(np.float32)

    pos = np.searchsorted(uk, sk)
    pos[pos == len(uk)] = 0
    ok = (uk[pos] == sk) & (counts[pos] * df_scale <= cap)
    sr, sf, pos = sr[ok], sf[ok], pos[ok]
    del sk, ok
    if verbose:
        print(f"[block] cap={cap} usable s1 keys={len(sr):,} expansion={counts[pos].sum():,}", flush=True)

    out = []
    # chunk S1 records so each chunk expands to <= budget pairs (bounded memory)
    rec_exp = np.bincount(sr, weights=counts[pos], minlength=len(s1))
    cum = np.cumsum(rec_exp)
    rb = np.unique(np.r_[0, np.searchsorted(cum, np.arange(budget, cum[-1] + budget, budget)), len(s1)])
    bounds = np.searchsorted(sr, rb)
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        if b1 <= b0:
            continue
        r, f, p = sr[b0:b1], sf[b0:b1], pos[b0:b1]
        cnt = counts[p]
        tot = int(cnt.sum())
        rep_idx = np.repeat(np.arange(len(r)), cnt)
        offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(starts[p], cnt)
        pairs = (r[rep_idx].astype(np.int64) << 24) | pr[offs].astype(np.int64)
        w = idf[p][rep_idx]
        fam = f[rep_idx]
        del offs, rep_idx
        up, inv = np.unique(pairs, return_inverse=True)
        del pairs
        s_name = np.bincount(inv, weights=w * (fam == 0), minlength=len(up)).astype(np.float32)
        s_addr = np.bincount(inv, weights=w * (fam == 1), minlength=len(up)).astype(np.float32)
        s_cross = np.bincount(inv, weights=w * (fam == 2), minlength=len(up)).astype(np.float32)
        nkeys = np.bincount(inv, minlength=len(up)).astype(np.int16)
        g = (up >> 24).astype(np.int32)
        keep = (_group_rank(g, s_name + s_addr + s_cross) < k_total) | (_group_rank(g, s_name) < k_name) | \
               (_group_rank(g, s_addr) < k_addr)
        out.append(pd.DataFrame({"s1_idx": g[keep], "pool_idx": (up[keep] & ((1 << 24) - 1)).astype(np.int32),
                                 "bs_name": s_name[keep], "bs_addr": s_addr[keep], "bs_cross": s_cross[keep],
                                 "bs_nkeys": nkeys[keep]}))
        if verbose:
            print(f"[block] s1 rows {r[0]:,}-{r[-1]:,}: expanded {tot:,} -> uniq {len(up):,} -> kept {keep.sum():,}"
                  f" ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(out, ignore_index=True)
