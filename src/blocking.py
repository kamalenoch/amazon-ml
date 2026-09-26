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
EXTRA_KEYS = True
MORE_KEYS = False          # --extra-keys: joined-name suffix + name-pair x house number (set before KeyIndex)
SINGLE_KINDS = set("trpjaz")


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
        if MORE_KEYS:                          # survives injected leading tokens ('@paramounttrading', '>> sion...')
            nk.add(f"z{c}|{joined[-8:]}")
    sk = list(dict.fromkeys(sk))[:MAX_PAIR_TOKS]
    if EXTRA_KEYS:
        # typo-tolerant name keys: pairs of 4-char prefixes ('consulatscy' ~ 'consultancy')
        pf = list(dict.fromkeys(t[:4] for t in ntoks if len(t) >= 3))[:MAX_PAIR_TOKS]
        for i in range(len(pf)):
            for j in range(i + 1, len(pf)):
                a, b = (pf[i], pf[j]) if pf[i] < pf[j] else (pf[j], pf[i])
                nk.add(f"c{c}|{a}|{b}")
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
        if MORE_KEYS:                          # common names made specific by the house number
            npair = sk[:MAX_CROSS_NAME]
            for i in range(len(npair)):
                for j in range(i + 1, len(npair)):
                    a, b = (npair[i], npair[j]) if npair[i] < npair[j] else (npair[j], npair[i])
                    for n in nl:
                        xk.add(f"y{c}|{a}|{b}|{n}")
        if EXTRA_KEYS:
            nu = list(dict.fromkeys(nums.split()))[:4]
            for i in range(len(nu)):          # number pairs: '5534 10480' survives a replaced name
                for j in range(i + 1, len(nu)):
                    a, b = (nu[i], nu[j]) if nu[i] < nu[j] else (nu[j], nu[i])
                    ak.add(f"u{c}|{a}|{b}")
    return nk, ak, xk


def _keys_chunk(args):
    off, countries, names, addrs, nums = args
    recs, keys, fams = [], [], []
    for i, (c, n, a, u) in enumerate(zip(countries, names, addrs, nums)):
        for fam, ks in enumerate(record_keys(c.lower(), n, a, u)):
            for k in ks:
                recs.append(off + i); keys.append(k)
                # fam code: +0 name / +1 addr / +2 cross ; +4 if single-token key (t,r,p,j,a)
                fams.append(fam + (4 if k[0] in SINGLE_KINDS else 0))
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
    """rank (0 = best) of each row within its group g, by descending score.
    Single int64 argsort on (group, -score) instead of lexsort (~5x faster)."""
    v = np.asarray(score, np.float32)
    bits = v.view(np.int32).astype(np.int64)
    bits = np.where(bits < 0, np.int64(0x7FFFFFFF) - bits, bits)   # order-preserving for negatives
    key = (np.asarray(g, np.int64) << 32) | (np.int64(0xFFFFFFFF) - (bits + np.int64(0x80000000)) & np.int64(0xFFFFFFFF))
    order = np.argsort(key, kind="stable")
    gs = g[order]
    first = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    rank_sorted = np.arange(len(g)) - np.repeat(first, np.diff(np.r_[first, len(g)]))
    rank = np.empty(len(g), np.int32)
    rank[order] = rank_sorted
    return rank


class KeyIndex:
    """Sorted pool-key index + S1 key lists; build once, run blocking passes many times."""

    def __init__(self, s1, pool, cap=600, df_scale=1.0, n_jobs=None, verbose=True, cap_composite=None):
        import time
        t0 = time.time()
        pr, pk, pf = build_keys(pool, n_jobs)
        sr, sk, sf = build_keys(s1, n_jobs)
        if verbose:
            print(f"[block] keys pool={len(pk):,} s1={len(sk):,} ({time.time() - t0:.0f}s)", flush=True)
        order = np.argsort(pk)
        pk = pk[order]
        self.pr = pr[order]
        pf = pf[order]
        del order, pr
        newk = np.empty(len(pk), bool)
        newk[0] = True
        np.not_equal(pk[1:], pk[:-1], out=newk[1:])
        self.starts = np.flatnonzero(newk)
        uk = pk[self.starts]
        del pk, newk
        self.counts = np.diff(np.r_[self.starts, len(self.pr)]).astype(np.int64)
        self.idf = np.log((len(pool) + 1) / (self.counts + 1)).astype(np.float32)
        # per-record idf mass of usable keys per family (name/addr/cross) -> cosine-normalized scores:
        # summed idf favours verbose records over true matches with short / empty addresses
        self.pmass = np.zeros((len(pool), 3))
        for a0 in range(0, len(self.pr), 50_000_000):
            a1 = min(a0 + 50_000_000, len(self.pr))
            kr = np.searchsorted(self.starts, np.arange(a0, a1), side="right") - 1
            f = pf[a0:a1]
            capr = np.where(f >= 4, cap, cap_composite or cap)
            w = np.where(self.counts[kr] * df_scale <= capr, self.idf[kr], 0).astype(np.float64)
            for fam_k in range(3):
                self.pmass[:, fam_k] += np.bincount(self.pr[a0:a1], weights=w * ((f & 3) == fam_k), minlength=len(pool))
        self.pmass = self.pmass.astype(np.float32)
        del pf
        pos = np.searchsorted(uk, sk)
        pos[pos == len(uk)] = 0
        capv = np.where(sf >= 4, cap, cap_composite or cap)
        ok = (uk[pos] == sk) & (self.counts[pos] * df_scale <= capv)
        self.sr, self.sf, self.pos = sr[ok], (sf[ok] & 3).astype(np.int8), pos[ok]
        self.n_s1 = len(s1)
        self.smass = np.stack([np.bincount(self.sr, weights=self.idf[self.pos].astype(np.float64) * (self.sf == k),
                                           minlength=len(s1)) for k in range(3)], 1).astype(np.float32)
        del sk, ok, uk
        if verbose:
            print(f"[block] cap={cap} usable s1 keys={len(self.sr):,} "
                  f"expansion={self.counts[self.pos].sum():,} ({time.time() - t0:.0f}s)", flush=True)


def generate_candidates(s1, pool, cap=600, df_scale=1.0, k_total=30, k_name=10, k_addr=10, k_cross=0,
                        budget=60_000_000, n_jobs=None, verbose=True, prerank=None, s1_subset=None, index=None,
                        k_cos=0):
    """Returns DataFrame [s1_idx, pool_idx, bs_*] (row indices into s1/pool).
    index: a prebuilt KeyIndex (reuse across passes). s1_subset: bool mask of S1 rows to process.
    prerank: callable(DataFrame)->DataFrame applied to every chunk's kept pairs (learned pruning)."""
    import time
    t0 = time.time()
    ix = index or KeyIndex(s1, pool, cap, df_scale, n_jobs, verbose)
    pr, starts, counts, idf = ix.pr, ix.starts, ix.counts, ix.idf
    sr, sf, pos = ix.sr, ix.sf, ix.pos
    if s1_subset is not None:
        m = s1_subset[sr]
        sr, sf, pos = sr[m], sf[m], pos[m]

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
        n_nk = np.bincount(inv, weights=(fam == 0), minlength=len(up)).astype(np.float32)
        n_ak = np.bincount(inv, weights=(fam == 1), minlength=len(up)).astype(np.float32)
        n_xk = np.bincount(inv, weights=(fam == 2), minlength=len(up)).astype(np.float32)
        nkeys = np.bincount(inv, minlength=len(up)).astype(np.int16)
        g = (up >> 24).astype(np.int32)
        s_tot = s_name + s_addr + s_cross
        keep = (_group_rank(g, s_tot) < k_total) | (_group_rank(g, s_name) < k_name) | \
               (_group_rank(g, s_addr) < k_addr) | (_group_rank(g, s_cross) < k_cross)
        extra = {}
        if k_cos:
            pidx = (up & ((1 << 24) - 1)).astype(np.int64)
            ms, mp_ = ix.smass[g], ix.pmass[pidx]
            cos = s_tot / np.sqrt((ms.sum(1) + 1.0) * (mp_.sum(1) + 1.0))
            ncos = s_name / np.sqrt((ms[:, 0] + 1.0) * (mp_[:, 0] + 1.0))
            acos = s_addr / np.sqrt((ms[:, 1] + 1.0) * (mp_[:, 1] + 1.0))
            keep |= _group_rank(g, cos) < k_cos
            extra = {"bs_cos": cos[keep].astype(np.float32), "bs_ncos": ncos[keep].astype(np.float32),
                     "bs_acos": acos[keep].astype(np.float32)}
            del pidx, ms, mp_, cos, ncos, acos
        blk = pd.DataFrame({"s1_idx": g[keep], "pool_idx": (up[keep] & ((1 << 24) - 1)).astype(np.int32),
                            "bs_name": s_name[keep], "bs_addr": s_addr[keep], "bs_cross": s_cross[keep],
                            "bs_nkeys": nkeys[keep], "bs_nnk": n_nk[keep], "bs_nak": n_ak[keep],
                            "bs_nxk": n_xk[keep], **extra})
        if prerank is not None:
            blk = prerank(blk)
        out.append(blk)
        del inv, w, fam, s_name, s_addr, s_cross, n_nk, n_ak, n_xk, nkeys, g, s_tot
        if verbose:
            print(f"[block] s1 rows {r[0]:,}-{r[-1]:,}: expanded {tot:,} -> uniq {len(up):,} -> kept {len(out[-1]):,}"
                  f" ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(out, ignore_index=True)
