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
from .translit import to_ascii as unidecode   # own romanizer (no GPL)

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


def raw_tokens(s: str):
    if not s:
        return ()
    if not s.isascii():
        s = unidecode(s)
    return tuple(sorted(set(_TOK.findall(s.lower()))))


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


NUM_COLS = ["num_rel_best", "num_rel_first", "num_eq_frac", "num_soft_frac", "num_sub_frac", "num_n1", "num_n2"]


# ------------------------------------------------------------------ number difference structure (ERR-01)
NUMD_COLS = ["numd_state", "numd_logabs", "numd_pos", "numd_ndig", "numd_rel"]


def number_diff_feats(n1, n2):
    """First S1 house number vs the closest same-length pool number (normalized digits).
    Distractors sit next door (small |d|, last digits differ); true copies carry typos in any digit
    (|d| >= 100 with one wrong digit: P(true) = 0.98 on train). numd_state: 2 equal, 1 same-length diff,
    0 no same-length number, -1 missing."""
    a, b = n1.split(), n2.split()
    if not a or not b:
        return (-1.0, -1.0, -1.0, -1.0, -1.0)
    x = a[0]
    if x in b:
        return (2.0, 0.0, 0.0, 0.0, 0.0)
    same = [v for v in b if len(v) == len(x)]
    if not same or not x.isdigit():
        return (0.0, -1.0, -1.0, -1.0, -1.0)
    xi = int(x)
    v = min(same, key=lambda z: abs(int(z) - xi) if z.isdigit() else 10 ** 12)
    if not v.isdigit():
        return (0.0, -1.0, -1.0, -1.0, -1.0)
    d = abs(int(v) - xi)
    dp = [i for i, (c1, c2) in enumerate(zip(x, v)) if c1 != c2]
    return (1.0, float(np.log10(d + 1)), float(len(x) - dp[0]) if dp else 0.0, float(len(dp)), d / max(xi, 1))


# ------------------------------------------------------------------ one-letter edit fingerprint (FORENSICS-02)
_VOW = frozenset("aeiou")
EDIT_COLS = ["n", "min", "sum", "pos_min", "len_min"]


def edit_key(x, z):
    """(op, position, letter class, length) of a one-edit change x -> z (Damerau distance 1), else None.
    The generator's typos in true copies hit the middle of a word or swap two letters; its distractor
    edits insert/delete/replace the FIRST or LAST letter, swap vowels, or change 1-3 letter words
    (dev train: first-letter insert/delete P(true)=0.02, last-letter insert 0.10, middle 0.73-0.81)."""
    lx, lz = len(x), len(z)
    if abs(lx - lz) > 1 or x == z or not x.isalpha() or not z.isalpha():
        return None
    m = min(lx, lz)
    i = 0
    while i < m and x[i] == z[i]:
        i += 1
    if lx == lz:
        if x[i + 1:] == z[i + 1:]:
            op, a, b = "sub", x[i] in _VOW, z[i] in _VOW
            ch = "vv" if a and b else ("cc" if not a and not b else "vc")
            last = lx - 1
        elif i + 1 < lx and x[i] == z[i + 1] and x[i + 1] == z[i] and x[i + 2:] == z[i + 2:]:
            op, ch, last = "tr", "", lx - 2
        else:
            return None
    else:
        long_, short = (z, x) if lz > lx else (x, z)
        if short[i:] != long_[i + 1:]:
            return None
        op, c = ("ins" if lz > lx else "del"), long_[i]
        ch = "dup" if (i > 0 and long_[i - 1] == c) or (i + 1 < len(long_) and long_[i + 1] == c) else ("v" if c in _VOW else "c")
        last = lx if op == "ins" else lx - 1
    pos = "first" if i == 0 else ("last" if i >= last else "mid")
    return op, pos, ch, min(lx, 7)


EDIT_CTX = False           # --edits-ctx: separate statistics when the edit is the ONLY name difference
MORPH = False              # --morph: also 2-3 letter suffix/prefix appends ('indraksh' -> 'indrakshyn')


def _morph_key(x, z):
    """2-3 letter append/removal at either end of an alphabetic token of >= 3 letters, keyed by the letters
    themselves so the log-odds table can price each one (distractors append pseudo-suffixes like 'yn')."""
    if not (x.isalpha() and z.isalpha()) or min(len(x), len(z)) < 3 or not 2 <= abs(len(z) - len(x)) <= 3:
        return None
    L = min(len(x), 7)
    if len(z) > len(x):
        if z.startswith(x):
            return "suf", z[len(x):], "", L
        if z.endswith(x):
            return "pre", z[:len(z) - len(x)], "", L
    else:
        if x.startswith(z):
            return "dsuf", x[len(z):], "", L
        if x.endswith(z):
            return "dpre", x[:len(x) - len(z)], "", L
    return None


def _edits(a, b):
    """Edit keys of every (dropped S1 token, added pool token) pair at Damerau distance 1."""
    sa, sb = set(a), set(b)
    drop, add = sa - sb, sb - sa
    if not drop or not add:
        return []
    ks = [k for x in drop for z in add if abs(len(x) - len(z)) <= 1 for k in (edit_key(x, z),) if k]
    if MORPH:
        ks += [k for x in drop for z in add for k in (_morph_key(x, z),) if k]
    if EDIT_CTX:              # a lone one-letter change is the distractor's move; typos come with other noise
        c = "1" if len(drop) == 1 and len(add) == 1 else "m"
        ks = [(c + k[0],) + k[1:] for k in ks]
    return ks


def _edit_names(k):
    op, pos, ch, L = k
    return (f"{op}|{pos}|{ch}|{L}", f"{op}|{pos}", f"{op}|L{L}", f"{op}|{ch}")


class EditCounter:
    """Log-odds (true vs negative candidate) of one-letter edit classes; learned on the dictionary slice."""

    def __init__(self):
        self.p, self.n = Counter(), Counter()
        self.npos = self.nneg = 0

    def update(self, pairs_a, pairs_b, y):
        for a, b, t in zip(pairs_a, pairs_b, y):
            keys = {s for k in _edits(a.split(), b.split()) for s in _edit_names(k)}
            if t:
                self.npos += 1
                self.p.update(keys)
            else:
                self.nneg += 1
                self.n.update(keys)

    def result(self, min_count=20, prior=1.0):
        return {k: float(np.log((self.p[k] + prior) / (self.npos + prior)) - np.log((self.n[k] + prior) / (self.nneg + prior)))
                for k in set(self.p) | set(self.n) if self.p[k] + self.n[k] >= min_count}


def edit_feats(a, b, lo):
    """Count of one-edit token changes and their log-odds: min/sum over edits of the most specific known
    class, plus the (op, position) and (op, length) classes alone."""
    ks = _edits(a, b)
    if not ks:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    best, pos, ln = [], [], []
    for k in ks:
        fine, p_, l_, c_ = _edit_names(k)
        v = lo.get(fine)
        best.append(v if v is not None else lo.get(p_, 0.0))
        pos.append(lo.get(p_, 0.0))
        ln.append(lo.get(l_, 0.0))
    return (float(len(ks)), min(best), sum(best), min(pos), min(ln))


# ------------------------------------------------------------------ signed house-number delta (--numsign)
NUMS_COLS = ["nums_d", "nums_lo", "nums_sign", "nums_nun"]


def _closest_delta(n1, n2):
    """Signed delta (pool - S1) of the closest same-length pair of numbers the records do NOT share,
    and the count of unshared S1 numbers. number_diff_feats keeps only |d| of the first S1 number: the sign
    matters (US distractors shift the house number UP by +1..+5, +7, +9, +11, +13, +21; P(true) is 0.76-0.83
    at d = -1/-2 but 0.08-0.09 at d = +1/+2 on dev, name Jaccard >= 0.5)."""
    a, b = n1.split(), n2.split()
    if not a or not b:
        return None, 0
    sa, sb = set(a), set(b)
    ua = [x for x in a if x not in sb and x.isdigit()]
    ub = [v for v in b if v not in sa and v.isdigit()]
    best = None
    for x in ua:
        xi = int(x)
        for v in ub:
            if len(v) == len(x):
                dd = int(v) - xi
                if best is None or abs(dd) < abs(best):
                    best = dd
    return best, len(ua)


def _delta_key(d):
    return str(d) if abs(d) <= 50 else ("big+" if d > 0 else "big-")


class DeltaCounter:
    """Log-odds (true vs negative candidate) of the exact signed delta, per country with a pooled back-off."""

    def __init__(self):
        self.p, self.n = Counter(), Counter()
        self.npos = self.nneg = 0

    def update(self, pairs_a, pairs_b, y, countries):
        for a, b, t, ct in zip(pairs_a, pairs_b, y, countries):
            if t:
                self.npos += 1
            else:
                self.nneg += 1
            d, _ = _closest_delta(a, b)
            if d is None:
                continue
            k = _delta_key(d)
            keys = (f"{(ct or '').lower()}|{k}", f"|{k}")
            (self.p if t else self.n).update(keys)

    def result(self, min_count=20, prior=1.0):
        return {k: float(np.log((self.p[k] + prior) / (self.npos + prior)) - np.log((self.n[k] + prior) / (self.nneg + prior)))
                for k in set(self.p) | set(self.n) if self.p[k] + self.n[k] >= min_count}


def number_sign_feats(n1, n2, country, lo):
    d, nun = _closest_delta(n1, n2)
    if d is None:
        return (-99999.0, 0.0, 0.0, float(nun))
    k = _delta_key(d)
    v = lo.get(f"{country}|{k}")
    if v is None:
        v = lo.get(f"|{k}", 0.0)
    return (float(max(min(d, 10000), -10000)), v, float(np.sign(d)), float(nun))
