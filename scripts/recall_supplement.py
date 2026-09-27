"""Recall supplement: find true pairs that blocking never proposed, with a multilingual bi-encoder + the trained CE.

Blocking misses are ~0.005 of the score (India: pool names in Indic scripts, empty addresses, typos). For every
pool record that the final submission leaves UNMATCHED, embed 'name | address' with multilingual-e5 (reads
Devanagari / Gujarati / Telugu / Latin alike), retrieve its top-k S1 of the same country by cosine, drop pairs that
were already candidates (they were scored and rejected), score the rest with the trained cross-encoder and keep the
best-scoring S1 per pool record. The threshold is calibrated on the labelled train holdout with the SAME procedure:
queries = unowned pool records (decoys) + holdout-owned true copies the model did not predict, targets = ALL train
S1 of the country, evaluation = macro F0.5 of the holdout S1 with the additions. No gain on the holdout -> nothing
is added.

  python scripts/recall_supplement.py --data <dataset> --pred-work <dir: holdout_preds, train_report.json, model.pkl,
      test_scores.parquet> --submission <dir: matching_results.tsv + candidate_pairs.tsv> --ce-model <ce_model dir>
      --out <dir>"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.run import s1_hash
from ce_kaggle import CE

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3)


def read(path, cols):
    return pd.read_csv(path, usecols=cols, **KW)


def texts(df):
    return (df.business_name + " | " + df.business_address).tolist()


class Encoder:
    """multilingual-e5 sentence embeddings (mean pooling, L2-normalized, fp16 on GPU)."""

    def __init__(self, name, dev):
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name).to(dev).eval()
        if dev == "cuda":
            self.m = self.m.half()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, tx, bs=512, max_len=64):
        d = self.m.config.hidden_size
        out = torch.empty((len(tx), d), dtype=torch.float16 if self.dev == "cuda" else torch.float32, device=self.dev)
        order = np.argsort([len(t) for t in tx], kind="stable")          # similar lengths per batch: less padding
        for s in range(0, len(tx), bs):
            idx = order[s:s + bs]
            e = self.tok(["query: " + tx[i] for i in idx], truncation=True, max_length=max_len, padding=True,
                         return_tensors="pt").to(self.dev)
            h = self.m(**e).last_hidden_state
            msk = e["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = torch.nn.functional.normalize(((h * msk).sum(1) / msk.sum(1).clamp(min=1)).float(), dim=-1)
            out[torch.as_tensor(idx, device=self.dev)] = v.to(out.dtype)
        return out


def keyof(s, p):
    return (np.asarray(s, np.int64) << 24) | np.asarray(p, np.int64)


def isin_sorted(k, sorted_keys):
    pos = np.searchsorted(sorted_keys, k)
    pos[pos >= len(sorted_keys)] = 0
    return sorted_keys[pos] == k if len(sorted_keys) else np.zeros(len(k), bool)


def propose(enc, s1_tx, s1_ct, q_idx, q_tx, q_ct, cand_keys_sorted, k, target_mask=None, qchunk=512, subset=None):
    """For every query pool record: top-k S1 of its country by cosine, minus existing candidates (and, for the
    calibration, minus S1 outside target_mask). Returns (query idx, s1 idx, cosine) - several per query possible."""
    Q, S, C = [], [], []
    for ct in pd.unique(q_ct):
        si = np.flatnonzero((s1_ct == ct) & (subset if subset is not None else True))
        qi = np.flatnonzero(q_ct == ct)
        if not len(si) or not len(qi):
            continue
        E = enc([s1_tx[i] for i in si])
        log(f"  {ct}: {len(si):,} S1 embedded; {len(qi):,} queries")
        for a in range(0, len(qi), 200_000):                             # embed queries in slices (bounded memory)
            part = qi[a:a + 200_000]
            EQ = enc([q_tx[i] for i in part])
            for b in range(0, len(part), qchunk):
                sc = EQ[b:b + qchunk] @ E.T
                v, j = sc.topk(min(k, len(si)), dim=1)
                v, j = v.float().cpu().numpy(), j.cpu().numpy()
                qq = np.repeat(part[b:b + qchunk], v.shape[1])
                ss = si[j.ravel()]
                Q.append(q_idx[qq]); S.append(ss); C.append(v.ravel())
            del EQ
        del E
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    Q, S, C = (np.concatenate(x) if x else np.zeros(0) for x in (Q, S, C))
    Q, S = Q.astype(np.int64), S.astype(np.int64)
    keep = ~isin_sorted(keyof(S, Q), cand_keys_sorted)
    if target_mask is not None:
        keep &= target_mask[S]
    return Q[keep], S[keep], C[keep].astype(np.float32)


def ce_score(ce, s1_tx, pool_tx, S, Q, bs=512):
    if not len(S):
        return np.zeros(0, np.float32)
    return ce.predict([s1_tx[i] for i in S], [pool_tx[j] for j in Q], bs=bs)


def best_by_cos(Q, S, C):
    """One proposal per query pool record: its most similar S1 that was not already a candidate."""
    return best_per_pool(Q, S, C)


def best_per_pool(Q, S, P):
    o = np.lexsort((-P, Q))
    q = Q[o]
    first = np.r_[True, q[1:] != q[:-1]] if len(q) else np.zeros(0, bool)
    b = o[first]
    return Q[b], S[b], P[b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred-work", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--ce-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="intfloat/multilingual-e5-small")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="testing: cap queries per split")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = os.environ.get("CE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    rep = json.load(open(os.path.join(a.pred_work, "train_report.json")))["args"]
    M = pickle.load(open(os.path.join(a.pred_work, "model.pkl"), "rb"))
    thr = float(M.get("thr", 0.725))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ce_kaggle import resolve_model
    enc = Encoder(resolve_model(a.encoder), dev)
    ce = CE(a.ce_model, dev)
    log(f"device {dev}; encoder {a.encoder}; CE {a.ce_model}; GBDT threshold {thr}")

    # ============================================================ calibration on the train holdout
    tr = f"{a.data}/train"
    s1 = read(f"{tr}/train_source1.tsv", ["entity_id", "business_name", "business_address", "country"])
    pool = pd.concat([read(f"{tr}/train_source{k}.tsv", ["entity_id", "business_name", "business_address", "country"])
                      for k in (2, 3)], ignore_index=True)
    ids, pids = s1.entity_id.values, pool.entity_id.values
    s1_tx, pool_tx = texts(s1), texts(pool)
    s1_ct, pool_ct = s1.country.values, pool.country.values
    del s1, pool
    gt = read(f"{tr}/train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"])
    gt = gt.assign(pool=gt.matched_entity_ids.str.split(",")).explode("pool")
    gt = gt[gt.pool.str.len() > 0]
    owner = np.full(len(pids), -1, np.int64)
    gi, gj = pd.Index(ids).get_indexer(gt.source1_entity_id.values), pd.Index(pids).get_indexer(gt.pool.values)
    ok = (gi >= 0) & (gj >= 0)
    owner[gj[ok]] = gi[ok]
    T = np.bincount(gi[ok], minlength=len(ids)).astype(np.float64)
    del gt
    h = s1_hash(ids)
    pr_end = int(rep["pr_frac"] * 10_000); m_end = pr_end + int(rep["m_frac"] * 10_000)
    hcut = pr_end + (1 - rep["holdout"]) * (m_end - pr_end)
    H = (h >= hcut) & (h < m_end)
    hp = pd.read_parquet(os.path.join(a.pred_work, "holdout_preds.parquet"))
    cand_h = np.sort(keyof(hp.s1_idx.values, hp.pool_idx.values))
    o = np.lexsort((-hp.p.values, hp.pool_idx.values)); ps = hp.pool_idx.values[o]
    b = o[np.r_[True, ps[1:] != ps[:-1]]]
    sel = b[hp.p.values[b] >= thr]
    ps1, ppj = hp.s1_idx.values[sel], hp.pool_idx.values[sel]
    n = len(ids)
    tp0 = np.bincount(ps1, weights=(owner[ppj] == ps1).astype(float), minlength=n)
    np0 = np.bincount(ps1, minlength=n).astype(np.float64)
    predicted = np.zeros(len(pids), bool); predicted[ppj] = True
    q_mask = ~predicted & ((owner < 0) | H[np.maximum(owner, 0)] & (owner >= 0))
    q_idx = np.flatnonzero(q_mask)
    if a.limit:
        q_idx = np.random.default_rng(0).choice(q_idx, min(a.limit, len(q_idx)), replace=False)
    log(f"calibration: holdout S1 {int(H.sum()):,}; queries {len(q_idx):,} (unowned + missed holdout copies)")
    sub = None
    if a.limit:                                   # testing only: sample the targets too (keeps the owners of the queries)
        sub = np.random.default_rng(1).random(len(ids)) < min(1.0, 5 * a.limit / len(ids))
        sub[owner[q_idx][owner[q_idx] >= 0]] = True
    Q, S, C = propose(enc, s1_tx, s1_ct, q_idx, [pool_tx[i] for i in q_idx], pool_ct[q_idx], cand_h, a.k, target_mask=H, subset=sub)
    Q, S, C = best_by_cos(Q, S, C)
    tru = owner[Q] == S
    floor = float(np.quantile(C[tru], 0.01)) if tru.sum() >= 50 else -1.0      # keeps 99% of the true proposals
    keep = C >= floor
    log(f"calibration proposals on holdout S1 (new pairs, best per pool record): {len(Q):,}, true {int(tru.sum()):,}; "
        f"cosine floor {floor:.3f} keeps {int(keep.sum()):,} (true {int(tru[keep].sum()):,})")
    Q, S, C = Q[keep], S[keep], C[keep]
    P = ce_score(ce, s1_tx, pool_tx, S, Q)
    hs = np.flatnonzero(H)

    def f05(tp, npred):
        return np.where(npred > 0, 1.25 * tp / np.maximum(npred + 0.25 * T, 1e-9), (T == 0).astype(float))[hs].mean()

    base = f05(tp0, np0)
    res = []
    for t in np.round(np.arange(0.50, 0.991, 0.02), 3):
        m = P >= t
        tpa = tp0 + np.bincount(S[m], weights=(owner[Q[m]] == S[m]).astype(float), minlength=n)
        npa = np0 + np.bincount(S[m], minlength=n)
        res.append((t, f05(tpa, npa) - base, int(m.sum()), float((owner[Q[m]] == S[m]).mean()) if m.any() else 0.0))
        log(f"  thr {t:.2f}: holdout F0.5 gain {res[-1][1]:+.5f}  added {res[-1][2]:,} (precision {res[-1][3]:.3f})")
    best = max(res, key=lambda r: r[1])
    # robust choice: the highest threshold that keeps >= 80% of the best gain (test has more decoys than train)
    t_use = max(r[0] for r in res if r[1] >= 0.8 * best[1]) if best[1] > 0 else None
    log(f"holdout base F0.5 {base:.4f}; best gain {best[1]:+.5f} @ {best[0]}; using threshold {t_use}")
    json.dump(dict(base=base, grid=res, t_use=t_use, floor=floor), open(os.path.join(a.out, "supplement_calibration.json"), "w"))
    del s1_tx, pool_tx, owner, hp, Q, S, P

    # ============================================================ test
    te = f"{a.data}/test"
    s1 = read(f"{te}/test_source1.tsv", ["entity_id", "business_name", "business_address", "country"])
    pool = pd.concat([read(f"{te}/test_source{k}.tsv", ["entity_id", "business_name", "business_address", "country"])
                      for k in (2, 3)], ignore_index=True)
    ids, pids = s1.entity_id.values, pool.entity_id.values
    s1_tx, pool_tx = texts(s1), texts(pool)
    s1_ct, pool_ct = s1.country.values, pool.country.values
    del s1, pool
    sc = pd.read_parquet(os.path.join(a.pred_work, "test_scores.parquet"), columns=["s1_idx", "pool_idx"])
    cand_t = np.sort(keyof(sc.s1_idx.values, sc.pool_idx.values))
    del sc
    matched = np.zeros(len(pids), bool)
    sub = pd.read_csv(os.path.join(a.submission, "matching_results.tsv"), **KW)
    mp = sub.matched_entity_ids.str.split(",").explode()
    mp = mp[mp.str.len() > 0]
    matched[pd.Index(pids).get_indexer(mp.values)] = True
    q_idx = np.flatnonzero(~matched)
    if a.limit:
        q_idx = np.random.default_rng(0).choice(q_idx, min(a.limit, len(q_idx)), replace=False)
    log(f"test: {len(ids):,} S1, {len(pids):,} pool, unmatched pool records {int((~matched).sum()):,} (queries {len(q_idx):,})")
    sub = (np.random.default_rng(1).random(len(ids)) < min(1.0, 5 * a.limit / len(ids))) if a.limit else None
    Q, S, C = propose(enc, s1_tx, s1_ct, q_idx, [pool_tx[i] for i in q_idx], pool_ct[q_idx], cand_t, a.k, subset=sub)
    Q, S, C = best_by_cos(Q, S, C)
    keep = C >= floor
    log(f"test proposals (new pairs, best per pool record): {len(Q):,}; above cosine floor {int(keep.sum()):,}")
    Q, S, C = Q[keep], S[keep], C[keep]
    P = ce_score(ce, s1_tx, pool_tx, S, Q)
    add = pd.DataFrame({"s1_id": ids[S], "pool_id": pids[Q], "pc": P, "cos": C, "country": s1_ct[S]})
    add.to_parquet(os.path.join(a.out, "supplement_pairs.parquet"))            # ALL scored proposals (for later merges)
    if t_use is None:
        log("no holdout gain -> nothing added; submission unchanged")
        add = add.iloc[:0]
    else:
        add = add[add.pc >= t_use]
    log(f"adding {len(add):,} pairs; by country {add.country.value_counts().to_dict()}")
    extra = add.groupby("s1_id").pool_id.apply(list).to_dict()
    sd = os.path.join(a.out, "submission")
    os.makedirs(sd, exist_ok=True)
    for name, col in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
        with open(os.path.join(a.submission, name), encoding="utf-8") as f, open(os.path.join(sd, name), "w", encoding="utf-8") as g:
            g.write(f.readline())
            for line in f:
                s, _, m = line.rstrip("\n").partition("\t")
                if s in extra:
                    m = ",".join([x for x in m.split(",") if x] + extra[s])
                g.write(f"{s}\t{m}\n")
    log(f"DONE: {sd}")


if __name__ == "__main__":
    main()
