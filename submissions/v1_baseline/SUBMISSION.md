# Submission v1 (baseline) — documented settings

The exact v1 code no longer exists; later commits changed the Blocking
defaults in place. v1 is recorded here as settings, not as a tagged commit.
Reproduce it by running the v2 code with the Blocking settings below.

| Setting | Value |
| --- | --- |
| Blocking token df cap | 500 |
| Blocking relative score cut | 0.25 of query's best |
| Candidates per S1 (k) | 10 |
| Model | LightGBM binary classifier, 35 features (see `policy.json`) |
| Assignment threshold `t` | 0.63 |
| Assignment margin | 0.2 |

## Measured (train holdout, folds 3-4)

| Metric | Value |
| --- | --- |
| Blocking recall@10 (full train) | 0.869 |
| Macro-F0.5 | 0.9147 |
| Pair precision / recall | 0.9885 / 0.8344 |
| Singleton accuracy | 0.960 |
| False merges | 29,519 |

Official validator: PASS (also with `--check-ids`). Not uploaded to the leaderboard.

## Files (not in git — TSVs are gitignored)

- `candidate_pairs.tsv`, `matching_results.tsv` — 1,732,544 test S1 rows each + header
- `policy.json`, `validation.json` — committed
