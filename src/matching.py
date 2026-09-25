"""Pair classifier, set selection, and the competition metric."""
import numpy as np
import pandas as pd
import lightgbm as lgb

NON_FEATS = {"y", "s1_idx", "pool_idx", "fold"}

LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_bin=255, verbose=-1, num_threads=0)


def feat_cols(X):
    return [c for c in X.columns if c not in NON_FEATS]


def train_lgb(X, y, rounds=600, params=None, valid=None):
    p = dict(LGB_PARAMS, **(params or {}))
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    vs = [lgb.Dataset(valid[0], valid[1], reference=dtr)] if valid is not None else []
    cb = [lgb.log_evaluation(100)] + ([lgb.early_stopping(50, verbose=False)] if vs else [])
    return lgb.train(p, dtr, num_boost_round=rounds, valid_sets=vs, callbacks=cb)


def select(s1_idx, pool_idx, p, thr=0.5, one_to_one=True, min_top=None):
    """Pick matches. Each pool record goes to at most one S1 (its best-scoring one).
    `min_top`: if set, an S1 whose best candidate has p >= min_top keeps it even if < thr."""
    df = pd.DataFrame({"s1": s1_idx, "pool": pool_idx, "p": p})
    if one_to_one:
        best = df.groupby("pool").p.transform("max")
        df = df[df.p >= best]
        df = df.drop_duplicates("pool")
    keep = df.p >= thr
    if min_top is not None:
        top = df.groupby("s1").p.transform("max")
        keep |= (df.p >= top) & (df.p >= min_top)
    return df[keep][["s1", "pool", "p"]]


def f05_macro(pred_pairs, true_pairs, all_s1):
    """pred_pairs/true_pairs: DataFrames with columns s1, pool (ids or idx). all_s1: iterable of s1 keys."""
    all_s1 = pd.Index(pd.unique(np.asarray(all_s1)))
    tp = pred_pairs.merge(true_pairs, on=["s1", "pool"]).groupby("s1").size()
    npred = pred_pairs.groupby("s1").size()
    ntrue = true_pairs.groupby("s1").size()
    tp, npred, ntrue = (s.reindex(all_s1, fill_value=0).values.astype(float) for s in (tp, npred, ntrue))
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0.0)
        rec = np.where(ntrue > 0, tp / ntrue, 0.0)
        f = np.where(tp > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
    f = np.where((ntrue == 0) & (npred == 0), 1.0, f)
    return float(f.mean()), dict(precision=float(prec[npred > 0].mean()) if (npred > 0).any() else 0.0,
                                 recall=float(rec[ntrue > 0].mean()),
                                 singleton_acc=float(((npred == 0)[ntrue == 0]).mean()) if (ntrue == 0).any() else 1.0)
