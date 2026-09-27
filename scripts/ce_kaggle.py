"""Cross-encoder re-scoring on top of the GBDT (GPU job).

1. Train a multilingual pair classifier on train candidate pairs (T job caches), excluding the GBDT holdout S1.
   Positives, hardest negatives by pre-ranker score, random negatives. Raw 'name | address' texts.
2. Re-score the GBDT holdout pairs whose GBDT probability lies in the uncertain band; pick the blend weight and
   threshold that maximize holdout macro F0.5 (all holdout S1 by hash, blocking misses included).
3. Re-score the uncertain test pairs (F job test_scores.parquet), blend, select (best S1 per pool record + threshold),
   write matching_results.tsv; candidate_pairs.tsv is copied from the predict output.

  python scripts/ce_kaggle.py --data <dataset dir> --train-work <T work> --pred-work <predict work: model.pkl,
      holdout_preds.parquet, test_scores.parquet> --test-work <S work: test_norm_*> --cands <predict output dir>
      --out <dir> [--model intfloat/multilingual-e5-small] [--n-pos 400000 --n-hard 600000 --n-rand 200000]"""
import argparse
import json
import math
import os
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.getcwd())
from src.prep import read_tsv, load_gt, _read_arrow
from src.run import s1_hash, write_lists
from src.matching import select

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def texts(path_tsvs, ids):
    need = set(ids)
    out = {}
    for p in path_tsvs:
        for ch in pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3, chunksize=1_000_000,
                              usecols=["entity_id", "business_name", "business_address"]):
            ch = ch[ch.entity_id.isin(need)]
            out.update(zip(ch.entity_id, ch.business_name + " | " + ch.business_address))
    return out


def resolve_model(name):
    """HF id if downloadable (Internet on), else a local copy attached under /kaggle/input."""
    import glob
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"])
    except Exception as e:                                     # offline
        short = name.split("/")[-1]
        hits = [os.path.dirname(p) for p in glob.glob(f"/kaggle/input/**/{short}*/**/config.json", recursive=True)]
        hits += [os.path.dirname(p) for p in glob.glob(f"/kaggle/input/**/config.json", recursive=True) if short in p]
        if not hits:
            raise RuntimeError(f"cannot download {name} ({e}) and no local copy under /kaggle/input") from e
        return hits[0]


class CE:
    def __init__(self, name, device):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(device)
        self.dev = device
        emb = self.m.get_input_embeddings()
        for p in emb.parameters():                 # the multilingual vocabulary table dominates the parameters
            p.requires_grad = False

    def enc(self, a, b, max_len=80):
        e = self.tok(a, b, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return {k: v.to(self.dev) for k, v in e.items()}

    def fit(self, A, B, y, bs=128, lr=5e-5, epochs=1):
        from transformers import get_linear_schedule_with_warmup
        opt = torch.optim.AdamW([p for p in self.m.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
        steps = epochs * math.ceil(len(y) / bs)
        sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=self.dev == "cuda")
        lossf = torch.nn.BCEWithLogitsLoss()
        self.m.train()
        k = 0
        for ep in range(epochs):
            order = np.random.default_rng(ep).permutation(len(y))
            for s in range(0, len(y), bs):
                r = order[s:s + bs]
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
                    out = self.m(**self.enc([A[i] for i in r], [B[i] for i in r])).logits.squeeze(-1)
                    loss = lossf(out.float(), torch.tensor(y[r], dtype=torch.float32, device=self.dev))
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()
                if k % 1000 == 0:
                    log(f"  step {k}/{steps} loss {loss.item():.4f}")
                k += 1
        self.m.eval()

    @torch.no_grad()
    def predict(self, A, B, bs=512):
        out = np.empty(len(A), np.float32)
        for s in range(0, len(A), bs):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
                z = self.m(**self.enc(A[s:s + bs], B[s:s + bs])).logits.squeeze(-1)
            out[s:s + bs] = torch.sigmoid(z.float()).cpu().numpy()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-work", required=True)
    ap.add_argument("--pred-work", required=True)
    ap.add_argument("--test-work", required=True)
    ap.add_argument("--cands", default=None, help="dir with the predict job's candidate_pairs.tsv; if missing, it is "
                    "rebuilt from test_scores.parquet (the predict job writes both from the same candidate table)")
    ap.add_argument("--scores", nargs="*", default=None, help="dirs with test_scores.parquet (per-country predict jobs); "
                    "merged by max. Default: --pred-work")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--n-pos", type=int, default=400_000)
    ap.add_argument("--n-hard", type=int, default=600_000)
    ap.add_argument("--n-rand", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--band", type=float, nargs=2, default=[0.01, 0.99])
    ap.add_argument("--force-w", type=float, default=None, help="use this GBDT weight instead of the holdout choice (testing)")
    ap.add_argument("--load", default=None, help="saved ce_model dir: skip training (rerun after a later-stage failure)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = os.environ.get("CE_DEVICE") or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    rep = json.load(open(os.path.join(a.pred_work, "train_report.json")))["args"]
    M = pickle.load(open(os.path.join(a.pred_work, "model.pkl"), "rb"))
    method = M.get("method", ("threshold", M["thr"]))
    log(f"device {dev}; GBDT selection {method}; model {a.model}")

    # ---------------- train pairs
    s1 = _read_arrow(os.path.join(a.train_work, "train_norm_s1.parquet"))[["entity_id"]]
    pool = _read_arrow(os.path.join(a.train_work, "train_norm_pool.parquet"))[["entity_id"]]
    ids, pids = np.asarray(s1.entity_id.values, dtype=object), np.asarray(pool.entity_id.values, dtype=object)
    c = pd.read_parquet(os.path.join(a.train_work, "train_cands.parquet"), columns=["s1_idx", "pool_idx", "pr_score"])
    _, gtp = load_gt(os.path.join(a.data, "train", "train_ground_truth.tsv"))
    gi = pd.Index(ids).get_indexer(gtp.s1_id.values); gj = pd.Index(pids).get_indexer(gtp.pool_id.values)
    ok = (gi >= 0) & (gj >= 0)
    tk = np.unique((gi[ok].astype(np.int64) << 24) | gj[ok])
    y = np.isin((c.s1_idx.values.astype(np.int64) << 24) | c.pool_idx.values, tk).astype(np.int8)
    h = s1_hash(ids)
    pr_end = int(rep["pr_frac"] * 10_000); m_end = pr_end + int(rep["m_frac"] * 10_000)
    hcut = pr_end + (1 - rep["holdout"]) * (m_end - pr_end)
    hold = (h >= hcut) & (h < m_end)
    usable = ~hold[c.s1_idx.values]
    rng = np.random.default_rng(0)
    pos = np.flatnonzero(usable & (y == 1)); neg = np.flatnonzero(usable & (y == 0))
    pos = rng.choice(pos, min(a.n_pos, len(pos)), replace=False)
    ord_ = neg[np.argsort(-c.pr_score.values[neg])]
    hard = ord_[:a.n_hard]
    rnd = rng.choice(ord_[a.n_hard:], min(a.n_rand, len(ord_) - a.n_hard), replace=False)
    rows = np.concatenate([pos, hard, rnd])
    log(f"train pairs {len(rows):,} (pos {len(pos):,} hard {len(hard):,} random {len(rnd):,})")
    tr_dir = os.path.join(a.data, "train")
    if not a.load:
        T1 = texts([f"{tr_dir}/train_source1.tsv"], ids[c.s1_idx.values[rows]])
        T2 = texts([f"{tr_dir}/train_source2.tsv", f"{tr_dir}/train_source3.tsv"], pids[c.pool_idx.values[rows]])
    if a.load:
        ce = CE(a.load, dev)
        log(f"loaded trained model from {a.load}")
    else:
        A = [T1[ids[i]] for i in c.s1_idx.values[rows]]; B = [T2[pids[j]] for j in c.pool_idx.values[rows]]
        ce = CE(resolve_model(a.model), dev)
        ce.fit(A, B, y[rows].astype(np.float32), epochs=a.epochs)
        del A, B
        ce.m.save_pretrained(os.path.join(a.out, "ce_model")); ce.tok.save_pretrained(os.path.join(a.out, "ce_model"))
        log("saved ce_model")
    del c

    # ---------------- holdout: blend weight + threshold
    hp = pd.read_parquet(os.path.join(a.pred_work, "holdout_preds.parquet"))
    lo, hi = a.band
    band = np.flatnonzero((hp.p.values >= lo) & (hp.p.values <= hi))
    H1 = texts([f"{tr_dir}/train_source1.tsv"], ids[hp.s1_idx.values[band]])
    H2 = texts([f"{tr_dir}/train_source2.tsv", f"{tr_dir}/train_source3.tsv"], pids[hp.pool_idx.values[band]])
    pc = ce.predict([H1[ids[i]] for i in hp.s1_idx.values[band]], [H2[pids[j]] for j in hp.pool_idx.values[band]])
    log(f"holdout band pairs re-scored: {len(band):,} of {len(hp):,}")
    hs1 = np.flatnonzero(hold)
    kk = pd.Series(gtp.s1_id.value_counts()).reindex(ids[hs1]).fillna(0).values

    def f05(p, thr):
        d = pd.DataFrame({"s1": hp.s1_idx.values, "pool": hp.pool_idx.values, "p": p, "y": hp.y.values})
        d = d[d.p >= d.groupby("pool").p.transform("max")].drop_duplicates("pool")
        d = d[d.p >= thr]
        tp = d.groupby("s1").y.sum().reindex(hs1).fillna(0).values
        npred = d.groupby("s1").size().reindex(hs1).fillna(0).values
        return np.where((npred == 0) & (kk == 0), 1.0, 1.25 * tp / np.maximum(npred + 0.25 * kk, 1e-9)).mean()

    thrs = np.round(np.arange(0.5, 0.951, 0.025), 3)
    pg = hp.p.values.astype(np.float32)
    best = max(((f05(pg, t), 1.0, t) for t in thrs))
    log(f"holdout GBDT alone: F0.5 {best[0]:.4f} @ {best[2]}")
    for w in (0.3, 0.5, 0.7):
        pb = pg.copy(); pb[band] = w * pg[band] + (1 - w) * pc
        cand = max(((f05(pb, t), w, t) for t in thrs))
        log(f"holdout blend w={w}: F0.5 {cand[0]:.4f} @ {cand[2]}")
        best = max(best, cand)
    if a.force_w is not None:
        pb = pg.copy(); pb[band] = a.force_w * pg[band] + (1 - a.force_w) * pc
        best = max(((f05(pb, t), a.force_w, t) for t in thrs))
    log(f"chosen: GBDT weight {best[1]}, threshold {best[2]}, holdout F0.5 {best[0]:.4f}")
    json.dump(dict(f05=best[0], w=best[1], thr=best[2]), open(os.path.join(a.out, "ce_blend.json"), "w"))

    # ---------------- test
    ts = None
    for d in (a.scores or [a.pred_work]):
        x = pd.read_parquet(os.path.join(d, "test_scores.parquet"))
        if ts is None:
            ts = x
        else:
            assert len(x) == len(ts) and (x.s1_idx.values == ts.s1_idx.values).all(), "different candidate sets"
            ts["p"] = np.maximum(ts.p.values, x.p.values)
    log(f"test scores from {len(a.scores or [a.pred_work])} dir(s): {len(ts):,} pairs")
    t1 = _read_arrow(os.path.join(a.test_work, "test_norm_s1.parquet"))[["entity_id"]]
    tp_ = _read_arrow(os.path.join(a.test_work, "test_norm_pool.parquet"))[["entity_id"]]
    tids, tpids = np.asarray(t1.entity_id.values, dtype=object), np.asarray(tp_.entity_id.values, dtype=object)
    p = ts.p.values.astype(np.float32)
    band = np.flatnonzero((p >= lo) & (p <= hi))
    log(f"test band pairs to re-score: {len(band):,} of {len(p):,}")
    if best[1] < 1.0:
        te = os.path.join(a.data, "test")
        Q1 = texts([f"{te}/test_source1.tsv"], tids[ts.s1_idx.values[band]])
        Q2 = texts([f"{te}/test_source2.tsv", f"{te}/test_source3.tsv"], tpids[ts.pool_idx.values[band]])
        pct = ce.predict([Q1[tids[i]] for i in ts.s1_idx.values[band]], [Q2[tpids[j]] for j in ts.pool_idx.values[band]])
        p[band] = best[1] * p[band] + (1 - best[1]) * pct
        pd.DataFrame({"row": band, "pc": pct}).to_parquet(os.path.join(a.out, "ce_test_band.parquet"))
    pd.DataFrame({"s1_idx": ts.s1_idx.values, "pool_idx": ts.pool_idx.values, "p": p}).to_parquet(os.path.join(a.out, "test_scores_blend.parquet"))
    sel = select(ts.s1_idx.values, ts.pool_idx.values, p, best[2])
    n = write_lists(os.path.join(a.out, "matching_results.tsv"), tids, sel.s1.values, pd.array(tpids, dtype="string"),
                    sel.pool.values, "matched_entity_ids")
    cp = os.path.join(a.cands or "", "candidate_pairs.tsv")
    if a.cands and os.path.isfile(cp):
        shutil.copy(cp, os.path.join(a.out, "candidate_pairs.tsv"))
    else:
        write_lists(os.path.join(a.out, "candidate_pairs.tsv"), tids, ts.s1_idx.values, pd.array(tpids, dtype="string"),
                    ts.pool_idx.values, "candidate_entity_ids")
        log("rebuilt candidate_pairs.tsv from the test scores")
    log(f"DONE: {len(sel):,} matched pairs ({len(sel) / len(tids):.2f} per S1), {n.mean():.3f} of S1 with a match")


if __name__ == "__main__":
    main()
