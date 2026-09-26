"""Normalization fixes for a country that is absent from training (France), applied to that country's rows only.
Other countries keep the cached normalization, so the model trained on US/India sees unchanged inputs there.

1. Admin components: region / departement names that the normalizer does not know as 'states' stay in the address
   ('hauts france', 'pas calais', 'gironde'), and true copies swap region <-> departement. They are LEARNED from the
   data (no table): components that dominate the last comma field of S1 or pool addresses of that country
   (the FR-ADMIN-01 rule, LB-confirmed in our first pipeline), and are removed from every address component.
2. Extra legal / generic forms (cie, ets, etablissements, freres, associes, ei, eirl, earl, gaec, scea, sem) are
   recognized as legal forms during that country's name normalization.
3. (off by default) removing the record's own country token from names: it hides the strongest decoy marker
   (a name that GAINS its country word is a decoy 100% of the time in train); on France it added 64k pairs, 42% of
   them with an added 'France'."""
import re

import numpy as np
import pandas as pd

from . import normalize as N
from .prep import NORM_COLS, normalize_df, read_tsv

EXTRA_LEGAL = {"cie": "cie", "ets": "ets", "etablissements": "ets", "freres": "freres", "associes": "associes",
               "ei": "ei", "eirl": "eirl", "earl": "earl", "gaec": "gaec", "scea": "scea", "sem": "sem"}
_NONALNUM = re.compile(r"[^a-z0-9 ]+")


def comp_clean(c):
    return " ".join(_NONALNUM.sub(" ", N.to_ascii(c or "").lower().replace(".", " ")).split())


def learn_admin(s1_addr, pool_addr, min_share=0.005):
    last = lambda a: comp_clean(a.split(",")[-1]) if (a or "").strip() else ""
    a = pd.Series([last(x) for x in s1_addr]).value_counts(normalize=True)
    b = pd.Series([last(x) for x in pool_addr]).value_counts(normalize=True)
    both = pd.DataFrame({"s1": a, "pool": b}).fillna(0)
    keep = both[(both.s1 >= min_share) | ((both.pool >= min_share) & (both.s1 < both.pool / 10))]
    return {k for k in keep.index if k and not any(ch.isdigit() for ch in k)}


def strip_admin(addr, admin):
    return ",".join(p for p in (addr or "").split(",") if comp_clean(p) not in admin)


def strip_country(name, country):
    out = re.sub(rf"\(?\b{re.escape(country)}\b\)?", " ", name or "", flags=re.I)
    out = " ".join(out.split())
    return out if re.search(r"[A-Za-z0-9]", out) else name


def _replace(frame, raw, cols, n_jobs):
    """Re-normalize the rows of `raw` (entity_id, business_name, business_address, ...) and write NORM_COLS into
    `frame` at the matching entity_id positions. Formatting columns (fn, fa, hn, ha) keep the ORIGINAL raw values."""
    pos = pd.Index(frame.entity_id.values).get_indexer(raw.entity_id.values)
    ok = pos >= 0
    raw, pos = raw[ok], pos[ok]
    saved = dict(N.LEGAL)
    N.LEGAL.update(EXTRA_LEGAL)                   # visible to the forked normalization workers
    try:
        nz = normalize_df(raw[["entity_id", "business_name", "business_address"]].copy(), n_jobs)
    finally:
        N.LEGAL.clear()
        N.LEGAL.update(saved)
    for c in cols:
        v = frame[c].to_numpy(dtype=object)
        v[pos] = nz[c].to_numpy(dtype=object)
        frame[c] = pd.array(v, dtype="string[pyarrow]")
        del v
    return int(ok.sum())


def apply_bundle(data_dir, split, s1, pool, countries, n_jobs=None, log=print, strip_country_token=False):
    """In-place: re-normalize the rows of `countries` in the cached s1 / pool frames with the bundle."""
    d = f"{data_dir}/{split}"
    raw1 = read_tsv(f"{d}/{split}_source1.tsv")
    rawp = pd.concat([read_tsv(f"{d}/{split}_source{k}.tsv") for k in (2, 3)], ignore_index=True)
    for ct in countries:
        r1, rp = raw1[raw1.country == ct].copy(), rawp[rawp.country == ct].copy()
        if not len(r1):
            log(f"[france] {ct}: no S1 rows, skipped")
            continue
        admin = learn_admin(r1.business_address.tolist(), rp.business_address.tolist())
        log(f"[france] {ct}: {len(admin)} learned admin components, e.g. {sorted(admin)[:12]}")
        for r in (r1, rp):
            r["business_address"] = [strip_admin(x, admin) for x in r.business_address]
            if strip_country_token:
                r["business_name"] = [strip_country(x, ct) for x in r.business_name]
        cols = [c for c in NORM_COLS if c in s1.columns]
        n1 = _replace(s1, r1, cols, n_jobs)
        n2 = _replace(pool, rp, cols, n_jobs)
        log(f"[france] {ct}: re-normalized {n1:,} S1 and {n2:,} pool rows")
    return s1, pool
