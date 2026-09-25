# EDA findings (train)

## S1 + ground truth
- S1: 2,206,821 rows (US 1,323,633 / India 883,188). No empty names/addresses.
- Singleton rate 5.6% (same for US and India). Matches per S1: mean 3.46, mode 3, max 11.
  Distribution 0:5.6%, 1:5.4%, 2:17%, 3:24%, 4:22%, 5:15%, 6:7.5%, 7+:4%.
- Per S1: S2 matches 0–5, S3 matches 0–6 -> S2/S3 contain within-source duplicates.
- 7,638,365 matched S2/S3 IDs, each belonging to exactly ONE S1 -> hard one-to-one constraint.
- 30% of S1 names are duplicates (case-insensitive) -> address is required to disambiguate.
- Postcodes essentially absent in S1.

## S2 / S3
| | S2 | S3 |
|---|---|---|
| rows | 5,034,616 | 5,285,603 |
| unmatched (distractors) | 26.6% (1.34M) | 25.4% (1.34M) |
| empty address | 3.4% | 3.3% |
| non-ASCII name (Indic scripts) | 15.2% | 11.5% |
| URL/domain-style name | 4.0% | 4.0% |
| ALL-CAPS name / address | 28% / 63% | 7.6% / ~0% |
| leading junk ("-- ") | 2.2% | 2.2% |

Noise observed:
- Names in Devanagari / Telugu / Bengali / Malayalam -> transliteration needed.
- Domain names instead of names ("ironl.com", "jarlolagold.com").
- OCR-style typos (PRlVATE, 0ne), typos (BRDNDS, MSEA, BIRMINGHAMM), injected tokens ("Center", "Services", "of"),
  word-order shuffles, suffix duplication ("Inc. Inc."), brackets ("(LLC)", "[LLC]"), "(ID: 20866)", "née", accents.
- Addresses: masked digits ("##570"), component reordering, full state name vs code, Indic-script state names,
  "CDP"/"CITY"/"Borough" suffixes, zero-padded numbers ("005905"), "S/O-" person refs, landmarks ("Near ...").

## Test
- S1 1,732,545 (India 809,986 / US 663,106 / **France 259,452**), S2 4,887,273, S3 5,082,316.
- France (unseen in train) noise: "R."/"BD."/"ALL." abbreviations, "N° 6", "Bis", "S.A.R.L.", departments
  (Gironde, Nord) instead of regions, injected accents ("Fôyer"), concatenated names ("CORALIEFETESSASCOM").
