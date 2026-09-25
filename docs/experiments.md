# Experiment log

Metric: macro F0.5 per S1 (singletons included). "Holdout" = 20% of a hashed 30% S1 sample of train,
scored against the full ground truth (blocking misses count as misses).

| Version | Date | Blocking recall | Cands / S1 | Holdout F0.5 | Thr | Public LB | Notes |
|---|---|---|---|---|---|---|---|
| dev | 2026-09-25 | 97.6% | 31.3 | 0.9663 | 0.70 | – | 5-state dev subset, df_scale=15.2 |
| **v1** | 2026-09-25 | 95.8% | 36.6 | **0.9597** | 0.70 | **0.9405** | full train, 6.0 h on Kaggle CPU |

## v1 details
- Holdout at thr 0.70: precision 0.987, recall 0.918, singleton accuracy 0.946.
- Test: 5,526,751 matched pairs; share of S1 with >=1 match: France 0.939, India 0.933, US 0.943.
- Top features (gain): bst_margin_pool, bst_rank_pool, bst_gap_pool, num_jacc, n_miss_idf_r, num_conflict, legal_eq.
- Runtime: normalize ~17 min, blocking 2 x ~42 min, string features 26 min (train), LGB fit ~55 min,
  test prediction ~2.7 h.

## Open questions
- LB (0.9405) is 0.019 below holdout. Hypothesis: France (15% of test, unseen) scores ~0.83.
  -> run the France diagnostic (`notebooks/diagnostic_france.py`).

## Ideas backlog
- Keep normalized + candidate caches as Kaggle output (skip ~2.5 h per run).
- Learned pre-ranker before top-K cut (dev ceiling 98.5% vs 97.6%).
- Expected-F0.5-optimal set selection per S1 instead of a global threshold.
- Stage 2 with context on stage-1 probabilities (+0.002 on dev), needs 2-fold OOF over all train S1.
- S2/S3 within-source duplicate clustering to propagate evidence.
- Faster inference: fewer trees / higher LR.
