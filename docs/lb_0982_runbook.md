# Public LB 0.982: how it was produced

Pipeline: **our v5 GBDT** + **e5-base cross-encoder** (re-scores the GBDT's uncertain pairs) + **recall supplement**
(finds true pairs that blocking never proposed). Key numbers are in `results/lb0.982/` (`summary.json`, `key_logs.txt`).

| Step | LB | What it added |
|---|---|---|
| v4.1b GBDT (fmt, cosine blocking, k_keep 40, number-difference, extra keys, one-letter edits) | 0.976 | – |
| + e5-small cross-encoder, blended in the GBDT band [0.01, 0.99] | 0.979 | pairs the GBDT got wrong (holdout +0.0010) |
| v5 GBDT (+ sound-key blocking/features) + **e5-base** cross-encoder + **recall supplement** | **0.982** | holdout: GBDT +0.0008, CE blend +0.0015, supplement +0.0032 |

## Notebooks, in order (all in `notebooks/`, run on Kaggle; "Save & Run All")

Every notebook embeds `src/` and its script, so it runs without the repo. Attach the outputs of earlier steps as inputs
("Add Input -> Your Work"); the notebooks find their inputs themselves and refuse mismatched ones.

| # | Notebook | Accelerator | Inputs | Time |
|---|---|---|---|---|
| 1 | `kaggle_T_train_v4` | CPU | challenge data, kaggle_b (wheels) | ~4 h |
| 2 | `kaggle_S_test_cands_v4` | CPU | challenge data, kaggle_b | ~3 h |
| 3 | `kaggle_R_resume_train_predict_v41b` | CPU | challenge data, kaggle_b, #1, #2 | ~3.5 h |
| 4 | `kaggle_CE_v41b` | GPU T4 x2, Internet | challenge data, kaggle_b, #1, #2, #3 | ~1 h |
| 5 | `kaggle_RS_v41b` | GPU T4 x2, Internet | challenge data, kaggle_b, #3, #4 | ~1.6 h |
| 6 | `kaggle_T5_train` | CPU | challenge data, kaggle_b | ~4.5 h |
| 7 | `kaggle_S5_test_cands` | CPU | challenge data, kaggle_b | ~3.5 h |
| 8 | `kaggle_F5_India` | CPU | challenge data, kaggle_b, #6, #7 | ~1 h |
| 9 | `kaggle_F5_US_France` | CPU | challenge data, kaggle_b, #6, #7 | ~1 h |
| 10 | `kaggle_CE_v5_base` | GPU T4 x2, Internet | challenge data, kaggle_b, #6, #7, #8, #9 | ~2 h |
| 11 | `kaggle_final` | CPU | challenge data, kaggle_b, #6, #8, #9, #10, #5 | ~6 min |

Submit `final/` of #11 (`matching_results.tsv` + `candidate_pairs.tsv`); it prints the validator result.
Steps 1-5 and 6-10 are independent and can run in parallel on separate accounts.

## Scripts
- `scripts/ce_kaggle.py`: cross-encoder: fine-tunes a multilingual e5 pair classifier on train candidates (GBDT holdout S1
  excluded), picks the GBDT/CE weight and threshold on the GBDT holdout, re-scores the test band. Saves `ce_holdout_band.parquet`,
  `ce_test_band.parquet`, the trained `ce_model/`; `--load` re-uses a trained model (inference only).
- `scripts/recall_supplement.py`: for every pool record the submission leaves unmatched: multilingual-e5 embedding, top-3 S1
  of the same country, drop existing candidates, score with the trained CE; threshold calibrated on the labelled holdout with
  the same procedure (nothing is added without a holdout gain).
- `scripts/final_assemble.py`: GBDT + 1-2 cross-encoders (weights/threshold on the holdout) + supplement pairs that are still
  new and unmatched -> submission; checks that every input comes from the same GBDT run.

## In progress on this branch (not part of 0.982)
`kaggle_CE_v5_large` (e5-large, 560M), `kaggle_LLM_v5` + `scripts/llm_rescore.py` (Qwen2.5-1.5B-Instruct + LoRA re-scorer),
`kaggle_RS_v5b` (supplement scored by the e5-base CE). The merge with main's v4/v5/v5.1 lives on a separate local branch.

## Rules
Models: MIT / Apache-2.0 only (multilingual-e5: MIT; Qwen2.5-1.5B-Instruct: Apache-2.0), all far below 8B parameters.
No external data, no APIs; the Kaggle notebooks download model weights only.
