"""Learned noise-model features.

The data generator perturbs true copies and distractors differently:
  * distractors ADD tokens from a specific vocabulary (public, private, group, holdings, india, ...)
    and SUBSTITUTE house-number digits (111 -> 115);
  * true copies add alias markers (dba, formerly, aka, nee), drop/mask/zero-pad digits (16282 -> 6282, ##1).
We learn per-token log-odds of being ADDED / DROPPED in true pairs vs. negative candidates (from
training labels, out-of-fold) and classify the edit relation between house numbers.
"""
import re
from collections import Counter

import numpy as np
from unidecode import unidecode

from .normalize import skeleton, collapse, LEGAL

SK = "\u00a7"   # prefix for phonetic-stem keys inside the log-odds dicts


from functools import lru_cache


@lru_cache(maxsize=2_000_000)
def stem_key(t: str):
    """Language-agnostic stem: singularize + consonant skeleton ('groupe'/'group' -> 'jrp',
    'developpement'/'development' -> 'dblpmnt'). None when too short to be distinctive."""
    if len(t) > 4 and t.endswith("s"):
        t = t[:-1]
    k = skeleton(collapse(t))
    return k if len(k) >= 3 else None


LEGAL_KEY, COUNTRY_KEY = SK + "LEGAL", SK + "COUNTRY"


def _special_key(t, country):
    if country and t == country:
        return COUNTRY_KEY
    if t in LEGAL and LEGAL[t]:
        return LEGAL_KEY
    return None


def lookup(lo, t, country="", freq=None):
    """exact token -> [only for corpus-frequent tokens] special class (own country name / legal form)
    -> phonetic stem -> 0. Frequency gate: distractor markers come from a vocabulary (frequent),
    typos are random (rare) and must not inherit a stem's statistics."""
    v = lo.get(t)
    if v is not None:
        return v
    if freq is None or t not in freq:
        return 0.0
    sp = _special_key(t, country)
    if sp:
        return lo.get(sp, 0.0)
    k = stem_key(t)
    return lo.get(SK + k, 0.0) if k else 0.0


_TOK = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"[0-9#]+")


_DOTTED = re.compile(r"(?<![a-z0-9])(?:[a-z]\.){2,}[a-z]?\.?")


def raw_tokens(s: str):
    if not s:
        return ()
    if not s.isascii():
        s = unidecode(s)
    s = s.lower()
    # dotted abbreviations are one token: 'E.U.R.L.' -> 'eurl', 'L.L.C.' -> 'llc' (not e/u/r/l letters)
    s = _DOTTED.sub(lambda m: m.group(0).replace(".", ""), s)
    return tuple(sorted(set(_TOK.findall(s))))


def raw_numbers(s: str):
    if not s:
        return ()
    return tuple(t for t in _NUM.findall(s) if any(c.isdigit() for c in t))


# ------------------------------------------------------------------ token log-odds
class LogOddsCounter:
    """Incremental version of learn_logodds (feed chunks, then .result())."""

    def __init__(self):
        self.add_p, self.add_n, self.drop_p, self.drop_n = Counter(), Counter(), Counter(), Counter()
        self.npos = self.nneg = 0

    def update(self, pairs_a, pairs_b, y, countries=None):
        countries = countries if countries is not None else [""] * len(y)
        for a, b, t, ct in zip(pairs_a, pairs_b, y, countries):
            sa, sb = set(a.split()), set(b.split())
            add, drop = sb - sa, sa - sb
            ct = (ct or "").lower()
            add = list(add) + [k for k in (_special_key(x, ct) for x in add) if k]
            drop = list(drop) + [k for k in (_special_key(x, ct) for x in drop) if k]
            if t:
                self.npos += 1
                self.add_p.update(add)
                self.drop_p.update(drop)
            else:
                self.nneg += 1
                self.add_n.update(add)
                self.drop_n.update(drop)

    def result(self, min_count=25, prior=1.0):
        npos, nneg = self.npos, self.nneg

        def lo(cp, cn):
            out = {k: float(np.log((cp[k] + prior) / (npos + prior)) - np.log((cn[k] + prior) / (nneg + prior)))
                   for k in set(cp) | set(cn) if cp[k] + cn[k] >= min_count}
            # phonetic-stem aggregate: lets unseen spellings in another language ('groupe') inherit
            # the statistics of their cognate ('group'); used only when the exact token is unknown
            sp, sn = Counter(), Counter()
            for c_, d_ in ((cp, sp), (cn, sn)):
                for k, v in c_.items():
                    if k.startswith(SK):
                        continue
                    sk = stem_key(k)
                    if sk:
                        d_[sk] += v
            for k in set(sp) | set(sn):
                if sp[k] + sn[k] >= min_count:
                    out[SK + k] = float(np.log((sp[k] + prior) / (npos + prior)) - np.log((sn[k] + prior) / (nneg + prior)))
            return out

        return lo(self.add_p, self.add_n), lo(self.drop_p, self.drop_n)


def learn_logodds(pairs_a, pairs_b, y, min_count=25, prior=1.0):
    """pairs_a/pairs_b: sequences of token tuples (S1 side, pool side). y: labels.
    Returns (add_lo, drop_lo) dicts: log P(token added | match) / P(token added | non-match)."""
    add_p, add_n, drop_p, drop_n = Counter(), Counter(), Counter(), Counter()
    npos = nneg = 0
    for a, b, t in zip(pairs_a, pairs_b, y):
        sa, sb = set(a), set(b)
        if t:
            npos += 1
            add_p.update(sb - sa)
            drop_p.update(sa - sb)
        else:
            nneg += 1
            add_n.update(sb - sa)
            drop_n.update(sa - sb)

    def lo(cp, cn):
        out = {}
        for k in set(cp) | set(cn):
            if cp[k] + cn[k] >= min_count:
                out[k] = float(np.log((cp[k] + prior) / (npos + prior)) - np.log((cn[k] + prior) / (nneg + prior)))
        return out

    return lo(add_p, add_n), lo(drop_p, drop_n)


def token_noise_feats(a, b, add_lo, drop_lo, country="", freq=None):
    sa, sb = set(a), set(b)
    add_t, drop_t = sb - sa, sa - sb
    add = [lookup(add_lo, t, country, freq) for t in add_t]
    drop = [lookup(drop_lo, t, country, freq) for t in drop_t]
    return (min(add, default=0.0), sum(add), max(add, default=0.0), float(sum(v < -2 for v in add)),
            min(drop, default=0.0), sum(drop), float(len(add_t)), float(len(drop_t)))


def frequent_tokens(series_list, min_df=50):
    c = Counter()
    for s_ in series_list:
        for v in s_:
            if v:
                c.update(v.split())
    return frozenset(k for k, n in c.items() if n >= min_df)


TOKEN_COLS = ["add_min", "add_sum", "add_max", "add_nbad", "drop_min", "drop_sum", "n_add", "n_drop"]

# ------------------------------------------------------------------ number edit relation
REL = ["equal", "zero_pad", "masked", "deletion", "insertion", "transpose", "substitution", "unrelated", "missing"]
_RANK = {r: i for i, r in enumerate(REL)}


def _rel(x, y):
    if x == y:
        return 0
    if x.lstrip("0") == y.lstrip("0"):
        return 1
    if "#" in y:
        core = y.replace("#", "")
        if (len(y) == len(x) and all(c == d or c == "#" for d, c in zip(x, y))) or (core and core.lstrip("0") in x):
            return 2
    lx, ly = len(x), len(y)
    if ly == lx - 1 and any(x[:i] + x[i + 1:] == y for i in range(lx)):
        return 3
    if ly == lx + 1 and any(y[:i] + y[i + 1:] == x for i in range(ly)):
        return 4
    if lx == ly:
        d = sum(c != e for c, e in zip(x, y))
        if d == 2 and sorted(x) == sorted(y):
            return 5
        if d == 1:
            return 6
    return 7


def number_feats(a, b):
    """a: S1 numbers, b: pool numbers. Returns best relation, counts of equal/substitution, coverage."""
    if not a or not b:
        return (8.0, 0.0, 0.0, 0.0, 0.0, float(len(a)), float(len(b)))
    best_for_a = [min(_rel(x, y) for y in b) for x in a]
    best = min(best_for_a)
    n_eq = sum(r <= 1 for r in best_for_a)
    n_soft = sum(2 <= r <= 5 for r in best_for_a)
    n_sub = sum(r == 6 for r in best_for_a)
    first = min(_rel(a[0], y) for y in b)
    return (float(best), float(first), n_eq / len(a), n_soft / len(a), n_sub / len(a), float(len(a)), float(len(b)))


def number_shift_feats(a, b):
    """Numeric alignment of S1 house/plot numbers to pool numbers. Distractors ('the business next
    door') shift the number by a small amount (4301->4306, 8159->8162, 4996->5003): 43% of distractors vs
    0.9% of true pairs have a 1-20 shift; single-digit-substitution rules miss most of these."""
    A = [int(x) for x in a if x.isdigit() and len(x) <= 9][:8]
    B = [int(x) for x in b if x.isdigit() and len(x) <= 9][:8]
    if not A or not B:
        return (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    ds = [min(abs(x - y) for y in B) for x in A]
    n = len(A)
    nz = [d for d in ds if d > 0]
    d0 = ds[0]
    return (sum(d == 0 for d in ds) / n,                       # share of S1 numbers found exactly
            sum(1 <= d <= 20 for d in ds) / n,                 # share shifted by a small amount
            float(min(nz)) if nz else 0.0,                     # smallest non-zero shift
            float(min(d0, 100000)),                            # shift of the first (house) number
            float(1 <= d0 <= 20),                              # house number shifted a little
            d0 / (A[0] + 1.0))                                 # relative shift


SHIFT_COLS = ["nshift_exact_frac", "nshift_small_frac", "nshift_min_nz", "nshift_first", "nshift_first_small",
              "nshift_first_rel"]

NUM_COLS = ["num_rel_best", "num_rel_first", "num_eq_frac", "num_soft_frac", "num_sub_frac", "num_n1", "num_n2"]
