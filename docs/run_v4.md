# Running v4 on Kaggle

v4 = v3 plus the dev-validated flags below. Dev holdout F0.5 goes from 0.9810 (v3) to 0.9861.
The notebooks already contain the code and the flags, so every job runs the same settings:

```
--fmt --k-keep 40 --cos 60 --numdiff --extra-keys --edits
```

The flags are generic (no France-specific rules). France-specific work (v3 stems and anything newer) is separate
and should be merged into this branch before the final submission.

## Jobs (Kaggle CPU, 4 vCPU / 30 GB, Accelerator: None)

| Job | Notebook | Inputs to attach | Est. time | Keeps |
|---|---|---|---|---|
| T | `notebooks/kaggle_T_train_v4.ipynb` | challenge data (+ wheels dataset if Internet is off) | ~2.5 h | `work/`: model.pkl, prerank.txt, train_cands, train_report.json, holdout_preds |
| S | `notebooks/kaggle_S_test_cands_v4.ipynb` | same | ~1.5 h | `work/`: test_norm_*, test_cands.parquet |
| F | `notebooks/kaggle_F_predict_v4.ipynb` | challenge data + **T output** + **S output** | ~1.5 h | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

Run T and S at the same time (two sessions or two accounts), then F.
`kaggle_pipeline_v4.ipynb` runs everything in one job instead (about 5 h).
Always use **Save Version → Save & Run All (Commit)**; only committed versions keep their output files.
Do not press Save Version on a notebook while it is running (ConcurrencyViolation).

## What to check in the logs

T (compare with v2 at the same step):

| log line | v2 | v4 expectation |
|---|---|---|
| `pre-ranker slice: ... wide recall` | 0.9777 | higher (cosine top-60 + extra keys) |
| `pre-ranker recall@40` | 0.9667 | closer to wide recall (cosine features in the pre-ranker) |
| `train cands ... blocking recall` | 0.9660 | higher |
| `edit classes: ...` | – | printed once; one-letter edit log-odds |
| `HOLDOUT F0.5` | 0.9784 | higher (dev: +0.0051) |

F: the validator cell must print `PASS`. `DONE: ... per S1` should be close to v2's 3.30.

## Memory

F builds the pair features one country at a time. No candidate crosses countries, so the features are
identical (dev check: bit-identical scores and outputs). The peak drops from ~29 GB (all ~69M test pairs × 52
columns at once, plus a copy) to ~20 GB. T peaks at an estimated ~21 GB (13M model rows × 121 features).
The validator cell looks for `validate_submission.py` in the challenge data (`utils/`) or any attached input.

## If something fails

- **Out of memory in T or S** (`Killed`, kernel restart): drop `--cos 60` and rerun that job (the cosine pass is the
  largest new memory user at full scale). Regenerate the notebook first (see below).
- **`feature order mismatch`** in F: T and F were built with different flags or code. Regenerate all three
  notebooks from the same commit and rerun T.
- **pip fails without Internet**: attach a dataset that contains `anyascii` and `rapidfuzz` wheels, or turn
  Internet on for the session.

## Regenerating the notebooks (after any change in `src/`)

```bash
export FLAGS="--fmt --k-keep 40 --cos 60 --numdiff --extra-keys --edits" SUFFIX=_v4
for st in train test-cands predict all; do STAGE=$st python scripts/make_notebook.py; done
```
