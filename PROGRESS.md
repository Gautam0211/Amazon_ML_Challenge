# PROGRESS — Amazon ML Challenge 2026, Business Entity Resolution

<!-- SESSION-STATE:BEGIN (overwritten at each milestone; resume from here) -->
## Improvement run — current state (2026-09-26 05:40)

- PRIMARY METRIC macro F0.5 (report F1, P, R, peak mem). User 09-26 05:00: SR1 NOT restarted; rerank + E007 SKIPPED.
  Remaining: e) Optuna F0.5 (RUNNING) -> confirm best on 3 folds full data, keep only if F0.5 > E006 0.96410
  f) threshold+margin (policy.py) on folds 0-2 | g) freeze, holdout folds 3-4 once, final.py + validate_lean.py.
  Keep submissions/E006_fallback as backup. If a job is killed: fall back, don't retry the same thing.
- gh: export PATH="$PATH:/c/Program Files/GitHub CLI". No Claude attribution in commits/PRs. Branch exp/p9-fallback.
- Best dev E006: F0.5 0.96410 F1 0.94622 P 0.98875 R 0.92664 peak 7.05G. Policy t=0.62 m=0.2 (coarse).
- Fallback E006 final: 5946s, peak commit 7.7G / RSS 4.47G, 93.05M test pairs, 5.72M matched. validate_lean PASS --check-ids.
- Rerank screen: +0.77pp blocking recall but ~4x blocking time; model screen killed (memory) -> skipped.
- Optuna: src/tune.py, study T2 in work/exp/optuna.db, log work/logs/tune_T2.out, timeout 9000s (ends ~07:53).
  Proxy: train folds 1-2 S1 frac 0.1 (25% of E006), eval 25% of fold-0 S1; min_data_in_leaf x4 for full data.
  T2 trial0 (E006) F05 0.96135 P 0.98860 R 0.91999 (t=0.70). Steady RSS 1.6G, build peak 6.05G.
  Confirm: python src/dev.py --exp E007 --parent E006 --params '<json>' --rounds <~iters> --cand cand_adB.parquet --sample_frac 0.4 --neg_keep 0.3 --groups str,ps
<!-- SESSION-STATE:END -->


Plain-language log. Only measured numbers appear here. "Pending" means not done yet.

## Dashboard

| Item | Value |
| --- | --- |
| Best measured macro-F0.5 (holdout) | **0.9602** (v2, folds 3-4; v1 was 0.9147) |
| Blocking candidate recall (train, all folds) | **0.9585** @ k=10 for v2 (ceiling macro-F0.5 0.985); v1 0.869 |
| Peak RAM (pipeline) | 3.9 GB process RSS (features stage) |
| Official validator status | **PASS** for v2 (final) and v1, both also with --check-ids |
| Submission ZIP | `TEAM_NAME_submission.zip` (363 MB, 18 files); rename it with the real team name |
| Leaderboard submissions made | 0 (none without owner approval) |

## Machine facts (measured 2026-09-25)

- RAM: 15.4 GB total, only ~7.2 GB free while other apps run → memory is the main constraint.
- CPU: 16 logical cores, no GPU. Disk D: ~127 GB free.
- Python 3.14 in project venv `D:\Amazon_Ml\.venv`.

## Stage 0 — Audit and the MemoryError

**What is it?** Checking what really exists before building more.

**What happened?** Data was extracted to `student_resource/dataset`. No pipeline code, model or output existed yet. Only two throwaway profiling scripts existed.

**The MemoryError, explained simply.** The profiling script asked pandas "which of these 7.6 million IDs are in this list of 10 million IDs?" (`Series.isin(set)`). In pandas 3, text columns are stored by pyarrow. For this call, pandas turned *every one* of the 7.6 million lookup values into a separate Python object (`pa.scalar`) before comparing. That object explosion ran out of memory. The IDs being long strings was a side issue; the real culprit was this specific operation.

**Fix.** Internally each record is identified by (source number, integer part of its ID). Verified on all 6 files: every ID has the exact form `S1-`/`S2-`/`S3-` + integer without leading zeros, so `"S2-" + str(number)` rebuilds the exact original ID (lossless; checked again by an assert in `prep.py`). Membership checks now run on integer numpy arrays, never on Python sets of strings.

## Data facts (Validation of assumptions)

Observed on training data (observations, not rules from the organisers, except where noted):

| Fact | Measured |
| --- | --- |
| Train rows S1 / S2 / S3 | 2,206,821 / 5,034,616 / 5,285,603 |
| Test rows S1 / S2 / S3 | 1,732,544 / 4,887,273 / 5,082,316 |
| Train countries | US, India. Test adds France (15% of test S1) — no labels for France exist |
| Duplicate IDs within a file | none |
| True matches per S1 | mean 3.46, median 3, max 11 |
| Singletons (S1 with no match) | 5.6% (123,247) |
| S2/S3 record linked to more than one S1 | 0 cases (observed) — supports "each S2/S3 goes to at most one S1" |
| S2/S3 records that match nothing (distractors) | ~26% |
| Country equal on every true pair | 100% |
| Names identical ignoring case | only 10.7% of true pairs → heavy noise |
| Non-ASCII names (native Indian scripts) | S1 0%, S2 15.2%, S3 11.5% |
| Empty address | S1 0%, S2/S3 ~3.4% |
| Test S2+S3 per S1 | ~5.8 (train ~4.7) → test pools are denser; a risk for tuned thresholds |

Noise seen: typos ("Cnetler"), reordered words, legal-suffix swaps (Pvt Ltd / Private Limited), DBA names, websites instead of names (`prudencegenesis.com`), unrelated random names with the same address, house numbers altered (6885 → 688), native script names (परफेक्ट इन्वेस्टमेंट्स), states in native script (తెలంగాణ).

## Stages

All code: `code/business_entity_resolution/src/`. All intermediate files: `work/` (Parquet, resumable). Log: `work/logs/run.log`.

### Stage 1+2 — Normalization and Transliteration  ✅ done (train + test)

**What is it?** Making text comparable. "Pvt. Ltd." and "Private Limited" should look the same; "परफेक्ट" (Hindi script) should become Latin letters ("prphekt").

**Why?** A computer compares characters. Without cleanup, the same business written two ways looks like two different businesses.

**What did we run?** `python src/prep.py --split train` and `--split test`.
- Transliteration: `anyascii` (offline, ISC licence) turns any script (Devanagari, Kannada, Tamil, Telugu, French accents) into ASCII.
- Names: lowercase, `&`→and, websites → their name part (`prudencegenesis.com` → `prudencegenesis`), phone numbers dropped, synonyms unified (pvt→private, ltd→limited ...). A "core name" drops legal words (LLC, Pvt, SARL, SAS ...).
- Phonetic skeleton: consonants only with similar sounds merged, so `marketimg`/`marketing` → `nrktng` and `praivet`/`private` → `prwt`. This bridges transliteration spelling differences.
- Addresses: abbreviations unified (Street/St, Road/Rd, Rue/R, Boulevard/Bd ...), numbers de-padded (`03520`→`3520`, `35th`→`35`), `<NULL>`/`N/A` removed.
- No rule looks at the country name, so France works the same way.

**Results.** Train: 3.8 min, peak RAM 2.1 GB. Test: 3.4 min, peak 2.2 GB. Lossless ID round-trip asserted for all 6 files.

**What broke and how was it fixed?** First run crashed with `WinError 1455: paging file too small`. Cause: 12 worker processes, each loading pandas/pyarrow, plus all data chunks pre-built at once → Windows "commit" memory (only ~5 GB free) ran out. Fix: 6 lightweight workers (no pandas import), chunks fed in small windows.

### Stage 3 — Blocking (candidate generation)  ✅ baseline done (train)

**What is it?** Comparing every S1 with every S2/S3 record would be 2.2M × 10.3M = 23 trillion pairs — impossible. Blocking quickly shortlists a few plausible S1 partners for each S2/S3 record.

**Why?** A true twin that is not on the shortlist can never be found later. So Blocking sets the maximum possible recall.

**How?** Each record becomes a bag of "tokens" (name words, joined name, phonetic skeletons, address words and numbers). Rare tokens get high weight (IDF). For every S2/S3 record we take the 10 S1 records with the highest weighted overlap (cosine). Only the same country label is compared (learned from data: 100% of true pairs share country; no country name is hard-coded). Tokens shared by more than 500 S1 records (like "street", "delhi") are not used to *create* pairs, so no block explodes.

**Results (train, measured):**

| Metric | Value |
| --- | --- |
| Candidate pairs | 93,507,908 (42.4 per S1) |
| Reduction ratio | 0.999996 |
| Recall@1 / @3 / @5 / @10 | 0.792 / 0.836 / 0.851 / **0.869** |
| Recall India / US | 0.857 / 0.877 |
| Runtime / peak RAM | 14.8 min / 3.8 GB |

**What broke?** First version (token cap 2000) needed ~50 min: sorting ~1,800 candidate scores per query was the bottleneck (measured). Fix: cap 500, drop scores below 25% of each query's best before sorting, and adaptive chunk sizes that halve when free RAM < 3 GB.

**Next.** 13% of true pairs are missed → biggest known ceiling. Will study the misses after the baseline is complete.

### Stage 4 — Feature Engineering  ✅ done (train: 93.5M pairs, 35 features, 24.3 min, peak 3.9 GB)

For each candidate pair: fuzzy name similarity (ratio, token-set, token-sort, partial, Jaro-Winkler) on full and core names; skeleton similarity; address similarity; house-number / all-number overlap; Jaccard overlaps; missing-address flags (−1 = unknown, not "mismatch"); blocking context (score, rank, gap to best / runner-up S1, how crowded the S1 is).

**What broke?** First run: `MemoryError` — the 93.5M-pair table was held twice (Arrow + numpy copies) plus 9 context features for all pairs (3.4 GB). Fix: one numpy copy, context features built per batch from small per-query/per-S1 arrays; batches of 250k pairs written as separate part files (resume = skip finished parts).

### Stage 5 — Model (LightGBM)  ✅ baseline done

**What is it?** LightGBM is a fast "decision tree ensemble". It looks at the 35 similarity numbers of a pair and outputs a probability that the two records are the same business.

**Why?** No single similarity is reliable (names get replaced by DBA names, addresses lose numbers). The model learns how to combine them from 7.6M labelled true pairs.

**What did we run?** `python src/model.py`. Training rows = all candidate pairs of a random 8% of S1 entities (7,356,876 pairs, 7.2% positive). Negatives are *hard negatives*: real Blocking candidates that look similar but are different businesses. 5 models, one per fold, each never sees the S1 entities it later scores (out-of-fold, grouped by S1 → no leakage). 400 trees, 127 leaves, lr 0.08, 12 threads.

**Results.** ~80 s per fold model. Most useful features (gain): gap between this S1 and the runner-up S1 for the same record, gap to the best S1, blocking rank, address token overlap, address token-set similarity, first house number equal, number overlap, name similarity. No country feature is used (helps France).

### Stage 6 — Assignment and Threshold  ✅ baseline done

**What is it?** The final decision rule. Each S2/S3 record is given to the single S1 it most likely belongs to, and only if the probability is at least a threshold `t` (and at least `margin` above the second-best S1).

**Why?** In training data no S2/S3 record belongs to two S1 entities (measured: 0 of 7.6M). Giving each record to only its best S1 removes many false merges.

**Measured (tune folds 0-2, official macro-F0.5):**

| Policy | t | margin | macro-F0.5 | pair precision | pair recall | singleton acc |
| --- | --- | --- | --- | --- | --- | --- |
| threshold only (no Assignment) | 0.70 | – | 0.9134 | 0.9890 | 0.8307 | 0.963 |
| Assignment | 0.30 | 0 | 0.9071 | 0.9689 | 0.8507 | 0.902 |
| Assignment | 0.50 | 0 | 0.9133 | 0.9824 | 0.8416 | 0.943 |
| Assignment | 0.65 | 0 | 0.9143 | 0.9885 | 0.8336 | 0.961 |
| **Assignment (chosen)** | **0.63** | **0.2** | **0.9145** | 0.9887 | 0.8338 | 0.961 |
| Assignment | 0.90 | 0 | 0.9064 | 0.9957 | 0.8062 | 0.985 |

The curve is flat between t = 0.55 and 0.75, so the choice is stable. Margin adds only +0.0002 (not meaningful). Assignment beats threshold-only by +0.001.

### Stage 7 — Validation  ✅ baseline done

**What is it?** Measuring the score honestly on S1 entities the model and the threshold search never saw.

**How?** 5 folds by S1 ID hash. Threshold chosen on folds 0-2, then reported once on folds 3-4 (881,793 S1 entities).

**HOLDOUT RESULT (baseline v1): macro-F0.5 = 0.9147.** Pair precision 0.9885, pair recall 0.8344, singleton accuracy 0.960, false merges 29,519, non-singleton F0.5 0.912.

**Where is the loss?** If the classifier were perfect on the current candidates, macro-F0.5 would be 0.938 (the Blocking ceiling). So: Blocking costs ~0.062, the model + Assignment cost ~0.023. **Blocking recall is the weakest stage** → next improvement target.

**France.** No labelled France data exists, so France cannot be validated. Mitigations: no country-specific rules or features, French legal forms and street words in the normalizer, country used only as "same label" in Blocking.

### Stage 8 — Output  ✅ v1 done

**What is it?** Writing the two official files for every one of the 1,732,544 test S1 records.

**What did we run?** `python src/block.py --split test`, `python src/features.py --split test`, `python src/predict.py`, then the official validator.

**Results (v1).**
- `output/matching_results.tsv`: 1,732,544 rows, 5,321,545 matched IDs, 92.2% of S1 have ≥1 match (train truth: 94.4% non-singleton — consistent).
- `output/candidate_pairs.tsv`: 90,797,293 candidate pairs = exactly the pairs the model scored; matches ⊆ candidates by construction.
- Per country: France 3.20 matches/S1 (6.3% empty), India 2.95 (9.3% empty), US 3.16 (6.7% empty). Manual check of 8 random France S1: all matches were the same business (spelling/accent/abbreviation variants). This is a sanity check, not a score.
- **Official validator: PASS**, also with `--check-ids` (every ID exists in the test files).
- Archived as `submissions/v1_baseline/`. **Not uploaded** — waiting for owner approval.

**What broke?**
1. Test features: workers died (`MemoryError`) at part 340/364 when other apps took RAM (free "commit" fell to 0.58 GB). Fix: killed the stuck run, restarted — finished parts were kept, only the rest was computed, with 2 workers. Workers are now recycled every 10 tasks.
2. Predict: building 91M ID strings at once needed ~5 GB (`ArrowMemoryError`). Fix: write the files line by line in S1 order; scores were already checkpointed, so no re-scoring.

## Experiments log (all measured)

| # | Change | Measured on | Result | Kept? |
| --- | --- | --- | --- | --- |
| E1 | Baseline v1 (cap 500, rel 0.25, k 10) | full train | recall@10 0.869, holdout macro-F0.5 **0.9147** | baseline |
| E2 | Where are true pairs lost? (holdout) | folds 3-4 | 13.1% never become candidates, 3.5% rejected by the model | → fix Blocking |
| E3 | Loose Blocking (cap 5000, rel 0, k 50) | 2% of queries | recall@10 **0.958**, @1 0.912 (vs 0.869 / 0.792) — but ~50× slower | insight: the df cap was dropping useful common tokens |
| E4 | cap 2000, rel 0.25 | 2% of queries | @10 0.932, 62 s | – |
| E5 | cap 5000, rel 0.4 | 2% of queries | @10 **0.958** (same as E3), 9.97 pairs/query, 78 s | **yes → v2** |
| E6 | cap 20000, rel 0.4 | 2% of queries | @10 0.972, but 316 s (~4 h per split) | not yet (too slow single-process) |
| E7 | Skeleton merges b/p, d/t, g/k (for Tamil/Telugu) | 2% of queries | non-ASCII @10 0.8977 vs 0.9020, overall 0.9578 vs 0.9583 | **no** (slightly worse; reverted) |
| E8 | Margin between best and runner-up S1 | tune folds | +0.0002 | kept (harmless) but not meaningful |
| E9 | v2 full train Blocking (cap 5000, rel 0.4, k 10) | full train | recall@10 **0.9585** (@1 0.912), India 0.936 / US 0.974, 28.3 pairs/S1, 49 min, peak 3.5 GB | running features/model on it |
| E10 | v2 LightGBM + Assignment (t 0.60, margin 0.2) | holdout folds 3-4 | macro-F0.5 **0.9602** (v1 0.9147); precision 0.987, recall 0.921, singleton acc 0.958, false merges 37,151; ceiling 0.985 | **yes → v2 is the final output** |

## Error analysis (v1 holdout)

- **False merges** (29,519 on holdout; pair precision 98.9%): mostly near-identical names at a *different* house/unit number in the same street ("1641 55th Ave" vs "1652A 55th Ave"; "H.No 8-2-14/53" vs "8-2-14/74"), or a sibling company with an extra word ("Diamond Educational … Group").
- **Missed matches**: (a) not in candidates — names replaced by unrelated trade names ("Umbraumbralum"), native-script names with partial addresses, city changed (Charlotte → Matthews); (b) rejected by the model — empty address on one side, or name reduced to one generic word.
- Non-ASCII (native script) records: Blocking recall 0.775 vs 0.884 for ASCII in v1 → transliteration is still the weakest text path.

## v2 run  ✅ done

**What:** the same pipeline with the looser Blocking (cap 5000, rel 0.4) for train and test (`work/run_v2.sh`, logs in `work/logs/v2_*.out`).

**Measured:**
- Train Blocking: 49 min, 62.5M pairs (28.3 per S1), recall@10 0.9585 (India 0.936, US 0.974).
- Features: 15 min.
- LightGBM + policy search: 19 min.
- Holdout macro-F0.5: **0.9602**.
- Test: Blocking 50 min, 60.3M pairs (34.8 per S1); predict 7 min.
- Output: 5.72M matched pairs; 94.0% of test S1 have at least one match (the train singleton rate is 5.6%).

**Problem:** test Feature Engineering crashed after 10 s with `MemoryError` / "OpenBLAS error: Memory allocation still failed". Each of the 4 worker processes reserved OpenBLAS thread buffers while the main process held the record strings, and together that used up the ~5 GB of free commit memory.

**Fix:** `OPENBLAS_NUM_THREADS=1`, `--workers 3`, `--batch 150000`. It then finished all 402 parts in 19.5 min (peak 3.0 GB). No finished work was lost, because every stage writes checkpoints.

**Output:** `output/*.tsv` are the v2 files (copies in `submissions/v2/`, and v1 in `submissions/v1_baseline/`). Official validator: PASS, also with `--check-ids`.

## Pending / owner actions
- Put the real team name and members into `Documentation_template.md`, then rebuild the ZIP: `python src/package.py --team <name>`.
- Upload to the competition portal. **Not done: this waits for owner approval.**
- Optional further gains (not run): a larger LightGBM training sample, a higher Blocking cap with parallel blocking, and a better native-script path.

## Leaderboard vs holdout (2026-09-25)
- Leaderboard v2: ~0.94. Holdout folds 3-4: ALL 0.9602 | India 0.9465 (ceil 0.9748) | US 0.9694 (ceil 0.9915). Per-country threshold re-tune: no gain (same t=0.60).
- Test mix 47% IN / 38% US / 15% FR -> implied France ~0.85-0.90. France is main gap; India second. Script: src/eval_country.py.

## France investigation (test set, no labels; diagnostics in work/fr_*.py)

- Coverage: 99.9% of France queries get candidates; predicted matches per S1 3.38 (US 3.36, India 3.23); predicted singletons 5.4%.
- Feature means of confident pairs (p>=0.95) match US/India (addr_tset 92.4 vs 91.9/93.9) → no sign of a France-specific breakdown.
- Only outlier: 0.9% of France queries have two S1 above 0.6 within the 0.2 margin (US/India 0.1%) → left unmatched. France S1 names repeat a lot ("bordeaux club" x530).
- Bugs found: "N°31" → token "ndeg31" (house number lost, 1.6% of France S2/S3); elision "l'Hopital" → "lhopital"; "cours"/"chem" not abbreviated. None occur in train data.
- Acronym names ("TA" = Tourcoing Atelier) are only 0.17% of train true pairs → not worth a feature.
- Conclusion: the France ≈ 0.85 figure was a back-calculation, not a measurement; evidence does not show France is broken.
