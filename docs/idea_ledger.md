# Idea ledger (all ideas, status, measured effect)

Dev = 5-state subset, holdout macro F0.5 (same protocol for every row). LB = public leaderboard.

## Facts that drive decisions
- Every country gets the same number of true matches per S1: 3.46 on average (train US and India identical).
- Test has ~1.9x the decoys per S1 of train (5.8 vs 4.7 pool records per S1; the extra ~1.1 per S1 have the
  decoy formatting profile). Dev cost of that density: 0.9865 -> 0.9849.
- Estimated LB loss split (v3.1, 0.9697): India ~0.013 (full-scale recall ~0.93), France ~0.009, US ~0.007.
- India losses before the matcher: common names whose keys are all over the frequency cap (91% on dev), made worse
  by native-script names romanized phonetically ('teknolajis', 'intaraneshanal'); US losses: name-only records of
  chains. About 80% of empty-address misses are irreducible (name shared by >= 2 S1).
- France (unseen): an unseen country loses mostly recall; ~60% of the gap is the noise dictionaries (India-unseen
  simulation: 0.8635, with India dictionaries 0.9376, fully seen ~0.984).

## Kept (in the current pipeline)
| idea | owner | effect |
|---|---|---|
| noise model (added/dropped token log-odds), pre-ranker, extra keys | teammate v2 | LB 0.941 -> 0.964 |
| French stem fallback for unseen tokens | teammate v3 | France predicted pairs 3.43 -> 3.15 per S1 |
| 2x model rows, holdout-chosen selection | teammate v3.1 | LB 0.9697 |
| format fingerprints, cosine top-60, keep 40, own romanizer + state fixes, number-difference, extra keys | v4 | dev 0.9810 -> 0.9853; full-scale blocking recall 0.966 -> 0.982, holdout 0.9784 -> 0.9826 |
| one-letter edit fingerprint (--edits) | v4 | dev +0.0008 |
| sound keys for native-script India names (--phon) | v5 | dev +0.0009; India native-script recall 0.954 -> 0.963 |
| France decoy-marker filter (post-processing) | reselect | LB 0.970 displayed; expected ~+0.0005 (check exact score) |
| decoy-density threshold (post-processing) | reselect | dev +0.0002 at 1.9x decoys |

## Tested, no gain (do not retry)
XGBoost / bigger LightGBM / blends (+-0.0006); expected-F0.5 selection (-0.0008); stage-2 on S1 context of p
(+0.0003); pool-margin and per-country threshold rules; at-least-one and second-chance rules (-0.0004);
ensembles of v3.1/v4/v5 (vote 0.9865, mean-p 0.9867 vs v5 0.9870); invented-name feature; lone-edit split;
reserved blocking slots; skeleton noise model (+0.0002); dictionary cap; numsign (+0.0001, optional);
suffix morphs (-0.0002); self-training for an unseen country (+0.002 on the country); pseudo-label
dictionaries (unstable -0.005..+0.006); training with decoys up-weighted (+0.0001); postcode split (no postcodes).

## Open (not yet tested in this pipeline)
| idea | where | expected | cost |
|---|---|---|---|
| France admin stripping (learned region/departement words) | normalization, France only | LB +0.0005..0.001 (was LB-confirmed in our old pipeline) | re-run S + predict France |
| missing French legal forms (cie, ets, etablissements, freres, associes, ei, eirl, earl, gaec) | normalization | small | same |
| drop the record's own country token from names ('(France)') | normalization | small | T + S + F |
| full-scale blocking diagnosis (notebook ready) -> targeted keys | blocking | informs the next step | 20 min Kaggle |
| reverse (pool-side) top-K for 'ranked out' losses | blocking | +0.0005 if full-scale diag shows many | T + S + F |
| refit on train + holdout after early stopping | training | +0.0002 | retrain |
| France exact twins: give a copy to both twins | post-processing | +0.0001 | minutes |
| LB probe: France rows emptied -> France's exact F0.5 | measurement | decides whether France work can pay | 1 submission |
