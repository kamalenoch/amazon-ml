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

    def update(self, pairs_a, pairs_b, y):
        for a, b, t in zip(pairs_a, pairs_b, y):
            sa, sb = set(a.split()), set(b.split())
            if t:
                self.npos += 1
                self.add_p.update(sb - sa)
                self.drop_p.update(sa - sb)
            else:
                self.nneg += 1
                self.add_n.update(sb - sa)
                self.drop_n.update(sa - sb)

    def result(self, min_count=25, prior=1.0):
        npos, nneg = self.npos, self.nneg

        def lo(cp, cn):
            return {k: float(np.log((cp[k] + prior) / (npos + prior)) - np.log((cn[k] + prior) / (nneg + prior)))
                    for k in set(cp) | set(cn) if cp[k] + cn[k] >= min_count}

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


def token_noise_feats(a, b, add_lo, drop_lo):
    sa, sb = set(a), set(b)
    add = [add_lo.get(t, 0.0) for t in sb - sa]
    drop = [drop_lo.get(t, 0.0) for t in sa - sb]
    return (min(add, default=0.0), sum(add), max(add, default=0.0), float(sum(v < -2 for v in add)),
            min(drop, default=0.0), sum(drop), float(len(sb - sa)), float(len(sa - sb)))


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
