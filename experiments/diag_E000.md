# Phase 0 diagnosis (E000), dev folds 0-2

Dev true pairs: 4,586,172; in candidates: 0.9583; missed by blocking: 191,150

## Blocking miss causes

| cause | pairs | share of misses | share of all true pairs |
|---|---|---|---|
| zero_overlap | 539 | 0.003 | 0.0001 |
| only_capped | 28,108 | 0.147 | 0.0061 |
| cutoff | 30,578 | 0.160 | 0.0067 |
| not_topk | 131,925 | 0.690 | 0.0288 |

Rank of true S1 (cap on, cutoff off, k unlimited) for `not_topk` misses:
| rank bucket | pairs |
|---|---|
| 10-20 | 30,379 |
| 20-50 | 37,169 |
| 50-100 | 19,685 |
| 100-inf | 44,687 |

True-score / best-score for `cutoff` misses:
| ratio | pairs |
|---|---|
| 0.3-0.4 | 15,235 |
| 0.2-0.3 | 11,703 |
| 0.1-0.2 | 3,603 |
| 0-0.1 | 37 |

## Slices: blocking recall and cause mix
| slice | value | share of true pairs | blocking recall | misses | zero_overlap | only_capped | cutoff | not_topk |
|---|---|---|---|---|---|---|---|---|
| nonascii | True | 0.138 | 0.9025 | 62,338 | 0.00 | 0.17 | 0.25 | 0.59 |
| country | India | 0.401 | 0.9354 | 118,766 | 0.00 | 0.13 | 0.21 | 0.66 |
| country | US | 0.599 | 0.9740 | 72,384 | 0.01 | 0.17 | 0.08 | 0.74 |
| q_addr_empty | True | 0.044 | 0.6923 | 62,378 | 0.01 | 0.20 | 0.06 | 0.73 |
| q_name_empty | True | 0.000 | 0.9821 | 32 | 0.09 | 0.12 | 0.16 | 0.62 |
| exact_name | True | 0.297 | 0.9903 | 12,989 | 0.00 | 0.02 | 0.00 | 0.98 |
| near_name | True | 0.447 | 0.9697 | 62,043 | 0.01 | 0.15 | 0.10 | 0.74 |
| weak_name_strong_addr | True | 0.108 | 0.9521 | 23,428 | 0.00 | 0.09 | 0.15 | 0.76 |
| no_name_no_addr_overlap | True | 0.025 | 0.6018 | 48,277 | 0.00 | 0.21 | 0.27 | 0.51 |
| src | S2 | 0.483 | 0.9603 | 87,797 | 0.00 | 0.16 | 0.16 | 0.68 |
| src | S3 | 0.517 | 0.9568 | 103,353 | 0.00 | 0.13 | 0.16 | 0.70 |

## Classifier misses (true pair in candidates, policy t=0.6 margin=0.2)
In-candidate true pairs: 4,395,022; predicted: 0.9608; missed: 172,473

| reason | pairs | share of all true pairs |
|---|---|---|
| lost_to_other_s1 | 33,281 | 0.0073 |
| below_threshold | 133,999 | 0.0292 |
| lost_on_margin | 5,193 | 0.0011 |

below_threshold by best-probability bucket:
| p | pairs |
|---|---|
| 0.5-0.6 | 26,879 |
| 0.4-0.5 | 23,643 |
| 0.3-0.4 | 22,434 |
| 0.1-0.3 | 40,604 |
| 0-0.1 | 20,439 |
