# Artifacts (GitHub Releases)

Large files are NOT in git (GitHub blocks files > 100 MB). Each pipeline version gets a GitHub Release
with its Kaggle outputs attached. Download from the repo's **Releases** page.

## Release `v1` (public LB 0.9405, holdout 0.9597)
| File | Size | What it is | Use it for |
|---|---|---|---|
| `matching_results.tsv` | 94 MB | Submitted predictions | Baseline / fallback submission |
| `candidate_pairs.tsv` | 841 MB | Blocking output (exact set scored by the model) | Try another matcher without re-running blocking |
| `test_scores.parquet` | 589 MB | `s1_idx, pool_idx, p` for all 63.5M test candidate pairs | Tune thresholds / selection / ensembles in minutes |
| `model.pkl` | 21 MB | `{model: LightGBM Booster, feats, thr, holdout_f05}` | Re-scoring; needs the same `src/` to build features |
| `train_report.json` | 1 KB | Holdout score, threshold, feature list | Reference |
| `kaggle_v1_log.txt` | – | Full notebook log | Timings, per-threshold table |

`s1_idx` = row index in `test_source1.tsv`; `pool_idx` = row index in `test_source2.tsv` followed by `test_source3.tsv`
(concatenated, S2 first).

### Load example
```python
import pickle, pandas as pd
M = pickle.load(open("model.pkl", "rb"))       # M["model"].predict(X[M["feats"]])
sc = pd.read_parquet("test_scores.parquet")
```

## Release `v2` (public LB 0.9640, holdout 0.9784)
| File | Size | What it is |
|---|---|---|
| `matching_results.tsv` | 96 MB | Submitted predictions (LB 0.963985) |
| `candidate_pairs.tsv` | 803 MB | Pre-ranked candidates (35 / S1) scored by the model |
| `run.log` | 8 KB | R job log (also in `results/v2/run_R_train_predict.log`) |
| `prerank.txt` | 1.4 MB | Pre-ranker LightGBM model (deterministic) |

Caches (`train_cands.parquet`, `test_cands.parquet`, `*_norm_*.parquet`, ~5.5 GB) live in the Kaggle dataset
`kamalenoch/s-and-t`; attach it to skip normalization and blocking.

## Release `v3.1` (public LB 0.969683) — current best
| File | Size | What it is |
|---|---|---|
| `v3.1_matching_results.zip` | 41 MB (96 MB unzipped) | Submitted predictions (LB 0.969683) |
| `v3.1_candidate_pairs.zip` | 355 MB (803 MB unzipped) | Candidates scored by the model (same as v2: 35 / S1) |

Run: R notebook (train + predict) on the v2 T and S caches. The run log and `model.pkl` of this run were not kept.
