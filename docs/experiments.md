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

## v4 candidates (branch feat/v4-improvements; dev only, not yet run at full scale)
Dev protocol: 5-state subset, `--df-scale 15.2 --m-frac 0.4 --holdout 0.5` (~30k holdout S1, 103.7k true pairs).
Matcher-only changes reuse the same cached candidates (paired comparison). LightGBM is deterministic here:
a reference rerun reproduces 0.9853 exactly. All changes are opt-in flags; defaults reproduce v3.

| Step (flags added) | Dev holdout F0.5 | Cand. recall | Kept |
|---|---|---|---|
| original v3 (4d672d2) | 0.9810 | 0.9811 | base |
| `--fmt --k-keep 40 --cos 60` + own romanizer / state fixes | 0.9845 | 0.9885 | yes |
| + `--numdiff` | 0.9853 | 0.9885 | yes |
| + `--extra-keys` | 0.9853 | 0.9892 | yes (targets full-scale common names) |
| + `--edits` | **0.9861** | 0.9892 | yes |
| + `--oov` (pool name made of words no S1 uses) | 0.9851 (vs 0.9853); with `--edits` 0.9860 | — | no |
| + `--edits-ctx` (edit statistics split: lone edit vs other differences) | 0.9858 (vs 0.9861) | — | no |
| + `--k-prior 10` (reserved wide slots: empty-address by name, invented-name by address) | 0.9860 | 0.9894 | no |
| + `--skelnoise` | +0.0002 | — | no |

Recommended v4 flags: `--fmt --k-keep 40 --cos 60 --numdiff --extra-keys --edits` (dev 0.9810 -> **0.9861**).
End-to-end check (test path on the dev files, model from the holdout run): validator PASS, macro F0.5 on the
model-holdout S1 0.9863. `--k-prior` barely moves wide recall (0.9899 -> 0.9900): the missed prior records share
no usable key with their S1, so reserved slots cannot reach them.

Learner comparison on the C_keys feature matrix (`--dump-xy`, same rows, threshold swept):
LightGBM (v3 params) 0.9848, LightGBM 255 leaves lr 0.05 0.9849, 3-seed LightGBM 0.9851, XGBoost depth 10 0.9854,
LightGBM+XGBoost 0.9853. **All within 0.0006: the features, not the learner, set the score.**
Expected-F0.5 per-S1 selection on the same probabilities: 0.9845 (-0.0008), confirming the v2 finding.
Post-hoc selection on fully scored dev candidates (tuned on half the holdout S1, reported on the other half):
pool-side margin rule -0.0002, soft penalty for a close second S1 +0.0001, per-country thresholds -0.0004.
Stage 2 with S1-side context of stage-1 p (2-fold OOF): 0.9860 alone, +0.0003 averaged with stage 1. Not kept:
the pool/S1 competition is already captured by the blocking/pre-ranker margin features.
Cross-script pool names: 14.8% of true pairs, 16.9% of false merges (miss rate 3.7% vs 2.9%): not a hotspot.

Findings behind the new flags (dev train candidates, labels from the dictionary/model slices only):
- **One-letter edits** (`--edits`, FORENSICS-02). For names that differ by one token at Damerau distance 1,
  P(true) depends on where the edit is: insert/delete of the FIRST letter 0.02, insert of the LAST letter 0.10,
  substitution of the last letter 0.14, vowel-for-vowel 0.33, 1-3 letter words 0.06-0.29; middle-of-word edits
  0.73-0.81 and letter swaps 0.85. Typos in true copies hit the middle; distractor edits hit the ends.
  Implemented like the token noise model: log-odds of (op, position, letter class, length) classes learned on the
  D slice for names and addresses; features = count, min, sum, (op,position) and (op,length) minima.
  False merges 463 -> 427, matcher misses 2,107 -> 2,027.
- **Empty pool address**: 99.3% of such records are true copies (non-empty: 76.3%), yet 30% of their true pairs
  were missed. 80% of those misses are irreducible: the name equals >= 2 S1 names (chains) and there is no
  address to break the tie. The model is calibrated there (P(true | only candidate) 0.93, p 0.935).
- **Invented names** ('Jaxhalo', 'Belodelta', domain/handle forms): 97-98% of pool records whose name is one
  word unseen in S1 are true copies; they are missed at 6.9% (2x average), half at blocking.
- House numbers: a pool number with one EXTRA digit is a distractor in every position (P(true) ~0.005);
  one deleted digit is weakly informative by position (first 0.72 vs last 0.64 for 4-digit numbers).
- No per-source exclusivity: an S1 has 0-5+ copies in S2 and in S3 independently.
- Remaining dev loss (C_keys): matcher misses +0.0070 if fixed, false merges +0.0041, blocking misses +0.0039.
  At full scale blocking is the larger term (train blocking recall 0.966 vs 0.989 on dev; pre-ranker
  recall@35 0.9654 vs @60 0.9699), so `--k-keep 40 --cos 60 --extra-keys` should matter more there than on dev.

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
