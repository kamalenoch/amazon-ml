# Experiment log

Metric: macro F0.5 per S1 (singletons included). "Holdout" = 20% of a hashed 30% S1 sample of train,
scored against the full ground truth (blocking misses count as misses).

| Version | Date | Blocking recall | Cands / S1 | Holdout F0.5 | Thr | Public LB | Notes |
|---|---|---|---|---|---|---|---|
| dev | 2026-09-25 | 97.6% | 31.3 | 0.9663 | 0.70 | – | 5-state dev subset, df_scale=15.2 |
| **v1** | 2026-09-25 | 95.8% | 36.6 | **0.9597** | 0.70 | **0.9405** | full train, 6.0 h on Kaggle CPU |
| v2-dev | 2026-09-25 | 98.1% | 33.5 | 0.9812 (M holdout) / **0.9837 full-pipeline on unseen S1** | 0.80 | – | dev subset |
| **v2** | 2026-09-26 | 96.6% | 34.9 | **0.9784** | 0.775 | **0.9640** | full train; T+S+R jobs; results/v2 |

## v1 details
- Holdout at thr 0.70: precision 0.987, recall 0.918, singleton accuracy 0.946.
- Test: 5,526,751 matched pairs; share of S1 with >=1 match: France 0.939, India 0.933, US 0.943.
- Top features (gain): bst_margin_pool, bst_rank_pool, bst_gap_pool, num_jacc, n_miss_idf_r, num_conflict, legal_eq.
- Runtime: normalize ~17 min, blocking 2 x ~42 min, string features 26 min (train), LGB fit ~55 min,
  test prediction ~2.7 h.

## v2 findings (dev, leak-free)
Loss decomposition of v1 (dev, 0.033 total): missed matches 46%, FPs 23%, true-but-predicted-empty 21%,
singleton FPs 10%. Model-missed pairs were mostly obvious matches with one perturbed field; FPs were 91%
distractor records (never matched to any S1).

The generator is learnable:
- Distractors ADD tokens from a fixed vocabulary: public (0 true / 862 neg), private (0/586), pvt (0/526),
  group, holdings, india, media, ventures ... True copies add alias markers (dba, formerly, aka, nee, fka).
- House numbers: digit SUBSTITUTION in 0.8% of true pairs vs 13.4% of FPs; deletion / masking / zero-padding
  in 7.6% of true pairs vs 2.7% of FPs.

| Change | Dev effect | Kept |
|---|---|---|
| Noise-model features (token add/drop log-odds learned out-of-sample + digit-edit relation) | 0.9655 -> 0.9814 | yes |
| Stage-2 sibling features | +0.0005 | no |
| Expected-F0.5 per-S1 selection | +0.0001 | no |
| Extra keys (number pairs, 4-char prefix pairs) | wide recall +0.3pp | yes |
| Cap 1200 single / 600 composite (vs 600/600) | wide recall +0.1pp, +12% expansion | yes |
| Learned pre-ranker, keep 35/S1 | recall@30 +0.4pp vs IDF-sum ranking | yes |
| rapidfuzz cpdist + preallocated features, arrow strings | ~10x faster features, ~3x less RAM | yes |

## v2 full-scale run (results/v2)
- Holdout at thr 0.775: precision 0.9967, recall 0.9454, singleton accuracy 0.9888.
- Blocking: train recall 96.6% (v1 95.8%); pre-ranker keeps 96.5% at K=35 vs 97.8% in the wide set
  (pruning costs 1.2pp at full scale; 0.4pp on dev).
- LB 0.9640 vs holdout 0.9784: gap 0.0144 (v1: 0.019). Holdout contains only US + India.
  If US/India test scored like holdout, France (15% of test) would be ~0.885.
- Share of S1 with >=1 match: France 0.949, India 0.935, US 0.942.
- Runtime: T 73 min to candidates; R (dicts + features + LGB + test) 116 min; peak RSS 18.5 GB.
- Engineering fixes on the way: OOM at 77M pairs (noise dicts as Python objects, parquet load),
  fixed with Arrow-native loading, streamed dictionaries, preallocated matrices.

## v2 -> v3: France (analysis on test outputs, no labels)
- France pool = 1.43M records for 259k S1 (5.5 per S1 vs 4.7 in train) -> ~37.5% distractors vs 26% in train.
- v2 predicted 3.43 matches / French S1 (US 3.35, India 3.22) -> implied French precision ~0.95 vs 0.997.
- French distractor markers: holding / distribution / international (~34k each, almost never matched -> model
  rejects: same spelling as English markers in the dictionary) but groupe (20.7k predicted pairs),
  developpement (13.3k), participations (9.2k) were ADDED tokens in predicted matches: dictionary keyed on exact
  tokens, 'group' = -8.9 and 'development' = -5.6 but 'groupe'/'developpement' unknown.
- In train, every added real word not in the S1 name has ~0 true pairs (frank 0/216, clinic 0/19088, group 0/63740):
  true noise only adds alias markers, legal suffixes, 'com', letters, typos.
- Fix (language-agnostic): unknown tokens fall back to (a) the record's own country name class (India 'india' = -8.71),
  (b) pooled legal-form class (-1.58), (c) phonetic stem (singular + consonant skeleton: groupe/group -> jrp),
  gated to corpus-frequent tokens (df >= 50) so random typos never inherit a stem's statistics.
- Rejected variant: stem-variant cancellation (+ ungated fallback): dev holdout 0.9809 -> 0.9796.
- Gated version: dev holdout 0.9815 (baseline 0.9809); flags 43.9k French predicted pairs (4.9%) vs 1.2k.

## v3.1 decisions (dev, fixed holdout)
- Training-size learning curve (fixed holdout): 0.44M rows 0.9788 | 0.88M 0.9804 | 1.77M 0.9814 -> gains halve per
  doubling; best_iter 182-303 (early stopping, not the round cap). Full run v2: 9.3M rows.
- Dictionaries need only the 15M-pair cap (~20% of train S1), so d_frac 0.55 -> 0.30 and m_frac 0.15 -> 0.30
  (2x model rows) at no dictionary cost; holdout share 0.20 -> 0.15.
- Expected-F0.5 per-S1 selection vs tuned global threshold: dev 0.9810 vs 0.9810 (and 0.9815 vs 0.9814 earlier).
  Both are now evaluated on every holdout and the better one is applied to test automatically.
- Holdout S1 set is defined by hash (includes S1 without candidates); scoring uses integer keys (seconds, not ~16 min).
- siblings.py (stage 2, +0.0005 on dev) moved to scripts/experiments/ - not part of the pipeline.

## v3.1 result
- Public LB 0.969683 (v2 0.963985, +0.0057). Only France-targeted changes and more model rows went in, on the same
  v2 candidates -> most of the LB-holdout gap was France.
- French predictions 3.43 -> 3.15 per S1; US / India unchanged.

## v4: decoding the distractor generator (dev, ground truth)
Structure: ~1.2 distractors per S1 (train: 2.68M distractors / 2.2M S1); 84% of S1 have a distractor as nearest
neighbour. 97.5% of model-missed true pairs already point to the right S1 -> the task is true-copy vs distractor.
Distractor = "the business next door": same name/street, house number shifted by a small amount and/or legal form
swapped / marker word added.

| Signal | True pairs | Distractors |
|---|---|---|
| house number shifted by 1-20 | 0.9% | 43.0% |
| house number equal | 80.8% | 35.0% |
| legal form swapped (LLC->Corp, Private->Public) | 0.26% | 14.8% |
| legal form added | 5.6% | 15.6% |

Candidate side: 97.7% of true pairs are their pool record's #1 S1, so each pool record's top-3 S1 (running top-k
across S1 chunks, pre-ranker score) is added to the S1-side top-35: recall 98.11% -> 98.50% for +3% pairs (dev).

| Change (dev, same holdout / same unseen S1) | Result |
|---|---|
| + number-shift alignment features + legal-transition flags | holdout 0.9810 -> 0.9825 |
| + pool-side top-3 candidates | holdout 0.9835 (expected-F selection now wins by +0.0003) |
| 255 leaves instead of 127 | 0.9827 (rejected) |
| Full pipeline on 7,503 never-seen S1: v2 / v3.1 / v4 | 0.9815 / 0.9810 / **0.9852** |

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
