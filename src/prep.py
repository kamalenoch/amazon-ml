"""Load the three sources, normalize in parallel, cache as parquet."""
import os
import multiprocessing as mp
import numpy as np
import pandas as pd

from .normalize import norm_name, norm_addr

READ_KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3)


def read_tsv(path):
    df = pd.read_csv(path, **READ_KW)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.replace("\r", "", regex=False)
    return df


def _norm_chunk(args):
    names, addrs = args
    out = []
    for n, a in zip(names, addrs):
        nc, _, legal = norm_name(n)
        ac, _, nums, st = norm_addr(a)
        out.append((nc, " ".join(legal), ac, " ".join(nums), st))
    return out


def normalize_df(df, n_jobs=None, chunk=50_000):
    n_jobs = n_jobs or os.cpu_count()
    names, addrs = df.business_name.tolist(), df.business_address.tolist()
    parts = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(df), chunk)]
    if n_jobs > 1 and len(parts) > 1:
        with mp.get_context("fork").Pool(n_jobs) as pool:
            res = pool.map(_norm_chunk, parts)
    else:
        res = [_norm_chunk(p) for p in parts]
    flat = [r for part in res for r in part]
    out = pd.DataFrame(flat, columns=["n_clean", "legal", "a_clean", "nums", "state"], index=df.index)
    # raw text is no longer needed downstream; dropping it roughly halves RAM at full scale
    return pd.concat([df.drop(columns=["business_name", "business_address"]), out], axis=1)


def load_split(data_dir, split, cache_dir=None, n_jobs=None):
    """Returns (s1, pool). pool = S2 ++ S3 with a `src` column (2/3)."""
    cache = os.path.join(cache_dir, f"{split}_norm") if cache_dir else None
    if cache and os.path.exists(cache + "_s1.parquet"):
        return pd.read_parquet(cache + "_s1.parquet"), pd.read_parquet(cache + "_pool.parquet")
    d = os.path.join(data_dir, split)
    s1 = read_tsv(os.path.join(d, f"{split}_source1.tsv"))
    s2 = read_tsv(os.path.join(d, f"{split}_source2.tsv"))
    s3 = read_tsv(os.path.join(d, f"{split}_source3.tsv"))
    s2["src"], s3["src"] = np.int8(2), np.int8(3)
    pool = pd.concat([s2, s3], ignore_index=True)
    s1["src"] = np.int8(1)
    s1 = normalize_df(s1, n_jobs)
    pool = normalize_df(pool, n_jobs)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        s1.to_parquet(cache + "_s1.parquet")
        pool.to_parquet(cache + "_pool.parquet")
    return s1, pool


def load_gt(path):
    gt = read_tsv(path)
    rows = [(s, m) for s, ms in zip(gt.source1_entity_id, gt.matched_entity_ids) for m in ms.split(",") if m]
    return gt, pd.DataFrame(rows, columns=["s1_id", "pool_id"])
