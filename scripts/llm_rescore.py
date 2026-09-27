"""LLM re-scorer: Qwen2.5-1.5B-Instruct (Apache-2.0) fine-tuned with LoRA as a pair classifier, applied to the pairs
that are STILL uncertain after GBDT + cross-encoder.

- Training pairs: train candidates of the T job (GBDT holdout S1 excluded): true copies + hardest decoys (pre-ranker rank).
- Narrow band: rows whose GBDT/CE blend (weight from ce_blend.json) lies in [lo, hi]; the most uncertain are scored first
  and a time budget bounds training and scoring (unscored rows stay missing - the final combiner handles that).
- LoRA is hand-written (q/k/v/o projections + the classification head), no peft dependency; base weights fp16 frozen,
  adapters fp32; both GPUs score in parallel.
- Output (same layout as the cross-encoder): llm_holdout_band.parquet / llm_test_band.parquet (row, pc) with rows indexing
  holdout_preds / the merged test scores, plus a holdout check (blend of the old score and the LLM on the narrow band).

  python scripts/llm_rescore.py --data <dataset> --train-work <T dir> --scores <F dir> [<F dir>] --ce <CE out dir> --out <dir>"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.getcwd())
from src.run import s1_hash

T0 = time.time()
KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=3)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def texts_for(paths, need):
    out = {}
    for p in paths:
        for ch in pd.read_csv(p, usecols=["entity_id", "business_name", "business_address"], chunksize=1_000_000, **KW):
            ch = ch[ch.entity_id.isin(need)]
            out.update(zip(ch.entity_id, (ch.business_name.str.slice(0, 120) + " | " + ch.business_address.str.slice(0, 160))))
    return out


def prompt(a, b):
    return f"Business A: {a}\nBusiness B: {b}\nSame business?"


class LoRALinear(torch.nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale


def build(model_dir, dev, r=16, alpha=32):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    m = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1)
    m.config.pad_token_id = tok.pad_token_id
    for p in m.parameters():
        p.requires_grad = False
    for mod in list(m.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear) and name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(mod, name, LoRALinear(child, r, alpha))
    head = getattr(m, "score", None) or getattr(m, "classifier")
    for p in head.parameters():
        p.requires_grad = True
    for p in m.parameters():   # frozen base fp16 on GPU (T4: no bf16), trainable adapters + head fp32 (checkpoint may load as bf16)
        if p.requires_grad:
            p.data = p.data.float()
        else:
            p.data = p.data.half() if dev.startswith("cuda") else p.data.float()
    return tok, m.to(dev)


def trainable_state(m):
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    return {k: v for k, v in m.state_dict().items() if k in names}


def enc(tok, A, B, dev, max_len):
    e = tok([prompt(a, b) for a, b in zip(A, B)], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    return {k: v.to(dev) for k, v in e.items()}


def fit(tok, m, A, B, y, dev, bs, lr, max_len, minutes):
    from transformers import get_linear_schedule_with_warmup
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = math.ceil(len(y) / bs)
    sch = get_linear_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    cuda = dev.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    lossf = torch.nn.BCEWithLogitsLoss()
    order = np.random.default_rng(0).permutation(len(y))
    m.train()
    t0 = time.time()
    for k, s in enumerate(range(0, len(y), bs)):
        r = order[s:s + bs]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cuda):
            z = m(**enc(tok, [A[i] for i in r], [B[i] for i in r], dev, max_len)).logits.squeeze(-1)
        loss = lossf(z.float(), torch.tensor(y[r], dtype=torch.float32, device=dev))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad(set_to_none=True)
        if k % 200 == 0:
            log(f"  step {k}/{steps} loss {loss.item():.4f} ({(s + len(r)) / max(time.time() - t0, 1e-9):.1f} pairs/s)")
        if time.time() - t0 > minutes * 60:
            log(f"  training time budget reached at step {k}/{steps} ({s + len(r):,} pairs)")
            break
    m.eval()


@torch.no_grad()
def predict(tok, models, A, B, bs, max_len, minutes=None):
    """Scores in the given order (callers put the most important first); rows not reached stay NaN."""
    out = np.full(len(A), np.nan, np.float32)
    t0 = time.time()
    batches = [np.arange(s, min(s + bs, len(A))) for s in range(0, len(A), bs)]
    for i in range(0, len(batches), len(models)):
        res = []
        for m, idx in zip(models, batches[i:i + len(models)]):
            dev = str(next(m.parameters()).device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.startswith("cuda")):
                res.append((idx, m(**enc(tok, [A[j] for j in idx], [B[j] for j in idx], dev, max_len)).logits.squeeze(-1)))
        for idx, z in res:
            out[idx] = torch.sigmoid(z.float()).cpu().numpy()
        if minutes is not None and time.time() - t0 > minutes * 60:
            log(f"  scoring time budget reached: {int(np.isfinite(out).sum()):,} of {len(A):,} rows scored")
            break
        if i % (200 * len(models)) == 0:
            log(f"  scored {int(np.isfinite(out).sum()):,}/{len(A):,}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-work", required=True, help="T dir: train_cands.parquet, holdout_preds.parquet, train_report.json")
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--ce", required=True, help="CE out dir: ce_holdout_band.parquet, ce_test_band.parquet, ce_blend.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-pos", type=int, default=50_000)
    ap.add_argument("--n-neg", type=int, default=50_000)
    ap.add_argument("--band", type=float, nargs=2, default=[0.05, 0.95])
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--score-bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--train-minutes", type=float, default=60)
    ap.add_argument("--score-minutes", type=float, default=75)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = os.environ.get("CE_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------------- ids, labels, holdout definition
    tr = f"{a.data}/train"
    ids = pd.read_csv(f"{tr}/train_source1.tsv", usecols=["entity_id"], **KW).entity_id.values
    pids = pd.concat([pd.read_csv(f"{tr}/train_source{k}.tsv", usecols=["entity_id"], **KW) for k in (2, 3)]).entity_id.values
    gt = pd.read_csv(f"{tr}/train_ground_truth.tsv", **KW)
    Tcount = pd.Series(gt.matched_entity_ids.str.count("S[23]-").values, gt.source1_entity_id.values).reindex(ids).fillna(0).values
    gt = gt.assign(pool=gt.matched_entity_ids.str.split(",")).explode("pool")
    gt = gt[gt.pool.str.len() > 0]
    owner = np.full(len(pids), -1, np.int64)
    gi, gj = pd.Index(ids).get_indexer(gt.source1_entity_id.values), pd.Index(pids).get_indexer(gt.pool.values)
    ok = (gi >= 0) & (gj >= 0)
    owner[gj[ok]] = gi[ok]
    rep = json.load(open(os.path.join(a.train_work, "train_report.json")))["args"]
    h = s1_hash(ids)
    pr_end = int(rep["pr_frac"] * 10_000); m_end = pr_end + int(rep["m_frac"] * 10_000)
    hcut = pr_end + (1 - rep["holdout"]) * (m_end - pr_end)
    hold = (h >= hcut) & (h < m_end)
    hs1 = np.flatnonzero(hold)

    # ---------------- training pairs (non-holdout S1): true copies + hardest decoys
    c = pd.read_parquet(os.path.join(a.train_work, "train_cands.parquet"), columns=["s1_idx", "pool_idx", "pr_score"])
    usable = ~hold[c.s1_idx.values]
    y = (owner[c.pool_idx.values] == c.s1_idx.values)
    rng = np.random.default_rng(0)
    pos = np.flatnonzero(usable & y)
    neg = np.flatnonzero(usable & ~y)
    pos = rng.choice(pos, min(a.n_pos, len(pos)), replace=False)
    neg = neg[np.argsort(-c.pr_score.values[neg], kind="stable")[:a.n_neg]]
    rows = np.concatenate([pos, neg])
    tr_s, tr_p, tr_y = c.s1_idx.values[rows], c.pool_idx.values[rows], y[rows].astype(np.float32)
    del c
    log(f"training pairs {len(rows):,} (true {int(tr_y.sum()):,}, hard decoys {int((tr_y == 0).sum()):,})")

    # ---------------- narrow band after the GBDT/CE blend
    blend = json.load(open(os.path.join(a.ce, "ce_blend.json")))
    w = float(blend["w"])
    hp = pd.read_parquet(os.path.join(a.train_work, "holdout_preds.parquet"))
    hb = pd.read_parquet(os.path.join(a.ce, "ce_holdout_band.parquet"))
    ph = hp.p.values.astype(np.float32).copy()
    ph[hb.row.values] = w * ph[hb.row.values] + (1 - w) * hb.pc.values
    lo, hi = a.band
    h_rows = np.flatnonzero((ph >= lo) & (ph <= hi))
    h_rows = h_rows[np.argsort(np.abs(ph[h_rows] - 0.5), kind="stable")]
    ts = None
    for d in a.scores:
        x = pd.read_parquet(os.path.join(d, "test_scores.parquet"))
        ts = x if ts is None else ts.assign(p=np.maximum(ts.p.values, x.p.values))
    tb = pd.read_parquet(os.path.join(a.ce, "ce_test_band.parquet"))
    pt = ts.p.values.astype(np.float32).copy()
    pt[tb.row.values] = w * pt[tb.row.values] + (1 - w) * tb.pc.values
    t_rows = np.flatnonzero((pt >= lo) & (pt <= hi))
    t_rows = t_rows[np.argsort(np.abs(pt[t_rows] - 0.5), kind="stable")]           # most uncertain first
    log(f"narrow band [{lo}, {hi}] after the blend (GBDT weight {w}): holdout {len(h_rows):,} rows, test {len(t_rows):,} rows")

    # ---------------- texts
    need1 = set(ids[np.concatenate([tr_s, hp.s1_idx.values[h_rows]])])
    needp = set(pids[np.concatenate([tr_p, hp.pool_idx.values[h_rows]])])
    X1 = texts_for([f"{tr}/train_source1.tsv"], need1)
    XP = texts_for([f"{tr}/train_source{k}.tsv" for k in (2, 3)], needp)
    te = f"{a.data}/test"
    tids = pd.read_csv(f"{te}/test_source1.tsv", usecols=["entity_id"], **KW).entity_id.values
    tpids = pd.concat([pd.read_csv(f"{te}/test_source{k}.tsv", usecols=["entity_id"], **KW) for k in (2, 3)]).entity_id.values
    Y1 = texts_for([f"{te}/test_source1.tsv"], set(tids[ts.s1_idx.values[t_rows]]))
    YP = texts_for([f"{te}/test_source{k}.tsv" for k in (2, 3)], set(tpids[ts.pool_idx.values[t_rows]]))
    log(f"texts loaded: train {len(X1):,}/{len(XP):,}, test {len(Y1):,}/{len(YP):,}")

    # ---------------- model
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(a.model):
        mdir = a.model
    else:
        from huggingface_hub import snapshot_download
        mdir = snapshot_download(a.model, allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
    tok, m = build(mdir, dev)
    ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    log(f"model {a.model}: trainable {ntr / 1e6:.1f}M params (LoRA + head), device {dev}")
    fit(tok, m, [X1[ids[i]] for i in tr_s], [XP[pids[j]] for j in tr_p], tr_y, dev, a.bs, a.lr, a.max_len, a.train_minutes)
    models = [m]
    if dev.startswith("cuda") and torch.cuda.device_count() > 1:          # second replica for scoring
        _, m2 = build(mdir, "cuda:1")
        m2.load_state_dict(trainable_state(m), strict=False)
        m2.eval()
        models.append(m2)
        log("scoring on 2 GPUs")

    # ---------------- holdout: does it help on the narrow band?
    pl_h = predict(tok, models, [X1[ids[i]] for i in hp.s1_idx.values[h_rows]], [XP[pids[j]] for j in hp.pool_idx.values[h_rows]],
                   a.score_bs, a.max_len)
    pd.DataFrame({"row": h_rows, "pc": pl_h}).to_parquet(os.path.join(a.out, "llm_holdout_band.parquet"))
    hs, hj, hy = hp.s1_idx.values, hp.pool_idx.values, hp.y.values.astype(float)
    n = len(ids)

    def f05(p, t):
        o = np.lexsort((-p, hj)); q = hj[o]
        b = o[np.r_[True, q[1:] != q[:-1]]]
        b = b[p[b] >= t]
        tp = np.bincount(hs[b], weights=hy[b], minlength=n); npr = np.bincount(hs[b], minlength=n)
        f = np.where(npr > 0, 1.25 * tp / np.maximum(npr + 0.25 * Tcount, 1e-9), (Tcount == 0).astype(float))
        return f[hs1].mean()

    thrs = np.round(np.arange(0.4, 0.951, 0.025), 3)
    base = max((f05(ph, t), t) for t in thrs)
    log(f"holdout GBDT+CE blend: F0.5 {base[0]:.4f} @ {base[1]}")
    best = (base[0], 0.0, base[1])
    for v in (0.25, 0.5, 0.75, 1.0):
        p2 = ph.copy()
        ok_ = np.isfinite(pl_h)
        p2[h_rows[ok_]] = (1 - v) * ph[h_rows[ok_]] + v * pl_h[ok_]
        f, t = max((f05(p2, t), t) for t in thrs)
        log(f"holdout + LLM (LLM share {v}): F0.5 {f:.4f} @ {t}")
        if f > best[0] + 2e-4 * (best[1] == 0.0):     # the LLM must beat 'no LLM' by >= 0.0002
            best = (f, v, t)
    log(f"LLM CHECK: best holdout F0.5 {best[0]:.4f} (LLM share {best[1]}, thr {best[2]}) vs {base[0]:.4f} without -> "
        f"{'HELPS' if best[1] > 0 else 'no gain'}")
    json.dump(dict(base=base[0], best=best[0], share=best[1], thr=best[2], band=[lo, hi], w=w,
                   holdout_rows=int(len(h_rows)), test_rows=int(len(t_rows))), open(os.path.join(a.out, "llm_check.json"), "w"))

    # ---------------- test (most uncertain first, within the time budget)
    pl_t = predict(tok, models, [Y1[tids[i]] for i in ts.s1_idx.values[t_rows]], [YP[tpids[j]] for j in ts.pool_idx.values[t_rows]],
                   a.score_bs, a.max_len, a.score_minutes)
    ok_ = np.isfinite(pl_t)
    pd.DataFrame({"row": t_rows[ok_], "pc": pl_t[ok_]}).to_parquet(os.path.join(a.out, "llm_test_band.parquet"))
    log(f"DONE: test rows scored {int(ok_.sum()):,} of {len(t_rows):,}")


if __name__ == "__main__":
    main()
