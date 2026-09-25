# Submission v2 (final) — git tag `v2`

Produced by the code at tag `v2` (`code/business_entity_resolution/`).

| Setting | Value |
| --- | --- |
| Blocking token df cap | 5000 |
| Blocking relative score cut | 0.4 of query's best |
| Candidates per S1 (k) | 10 |
| Model | LightGBM binary classifier, 35 features (see `policy.json`) |
| Assignment threshold `t` | 0.60 |
| Assignment margin | 0.2 |
| Test feature stage | `OPENBLAS_NUM_THREADS=1`, `--workers 3`, `--batch 150000` |

## Measured (train holdout, folds 3-4)

| Metric | Value |
| --- | --- |
| Blocking recall@10 (full train) | 0.9585 |
| Macro-F0.5 | 0.9602 |
| Pair precision / recall | 0.987 / 0.921 |
| Singleton accuracy | 0.958 |
| False merges | 37,151 |

Official validator: PASS (also with `--check-ids`). Same files as `output/`.

## Files (not in git — TSVs are gitignored)

- `candidate_pairs.tsv`, `matching_results.tsv` — 1,732,544 test S1 rows each + header
- `policy.json` — committed
