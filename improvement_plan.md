# ML Challenge 2026 — Entity Resolution Improvement Plan

> Hand-off document for Claude Code. Treat the current pipeline as a **frozen baseline**, improve it one measured change at a time, and never tune on folds 3–4.

---

## 0. Current Baseline (do not change until Phase 0 is done)

### Pipeline
1. Unicode / name / address normalization + transliteration + phonetic skeletons
2. Sparse IDF-weighted TF-IDF blocking (`block.py`)
3. Top-k candidate generation per S2/S3 query
4. 35 pairwise similarity/context features
5. LightGBM binary classifier
6. Threshold + margin + one-best-S1 assignment
7. Outputs: `matching_results.tsv`, `candidate_pairs.tsv`

### Blocking (v2 settings)
- Every S2/S3 record is compared against S1 records **of its own country only**; the country list is read from the data (France needed no special code).
- Hashed tokens: name words, name with spaces removed, phonetic skeletons (consonant patterns bridging transliteration/vowel typos), address words and numbers.
- IDF-weighted cosine via sparse matrix product (scipy); queries run in chunks sized to free RAM.
- `k = 10`, DF cap = 5,000 (a token in >5,000 S1s cannot create pairs on its own), relative cutoff = 0.4 × query's best score.
- Result: ~28 candidates/S1, blocking recall **0.9585** (up from 0.869 in v1), blocking ceiling **~0.985**.

### Model (manually fixed, never tuned)
`learning_rate=0.08, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=1.0, 400 trees`. Trained on blocking candidates from ~8% of S1.

### Current validation result
| Metric | Value |
|---|---|
| Macro F0.5 | 0.9602 |
| Precision | 0.987 |
| Recall | 0.921 |
| Blocking recall | 0.9585 |
| Blocking ceiling | ~0.985 |
| Native / non-ASCII recall | ~0.90 (vs ~0.96 overall) |

Prior threshold + margin + assignment tuning added only ~+0.0012, so decision-rule tuning is a small lever.

### What the numbers say
Precision is already 0.987; the gap is **recall**. ~4.15% of true pairs never reach the classifier. Blocking is the first bottleneck, not model complexity.

---

## 1. Hard Constraints

**Compute**
- CPU only, ~4–5 GB practical memory/commit limit.
- Use chunking, integer IDs, Parquet intermediates, resumable/cached stages, memory-safe processing.

**Compliance**
- Only the provided challenge data. No external lookups, APIs, geocoding, internet augmentation, external transliteration data, or LLM-based matching.

**Output validity (must hold for every run)**
- Every S1 appears exactly once.
- Predictions reference only S2/S3 IDs; no duplicate IDs.
- Every prediction in `matching_results.tsv` is a subset of `candidate_pairs.tsv`.
- `candidate_pairs.tsv` is exactly the candidate set fed into inference.
- Unseen countries (e.g. France) keep working — no hard-coded US/India logic.

**Validation protocol**
- Folds grouped by S1; no S1 leakage across folds.
- Folds 0–2: all development, tuning, feature selection, blocking selection, threshold selection.
- Folds 3–4: untouched holdout, evaluated **once** at the end.
- Never tune against the public leaderboard.

---

## 2. Metrics

- **Primary development metric:** entity-level (per-S1) **macro F0.5** — this is what the leaderboard scores. Computed with the full prediction pipeline (assignment + threshold + margin), averaged over folds 0–2. (Changed from F1 after E005; E000–E005 decisions stand because F0.5 rose with every kept change.)
- **Always also report:** F1-macro, precision, recall, singleton accuracy (correctly predicting "no match").
- **Blocking metrics:** recall, ceiling, candidates/S1, total candidate pairs.
- **Cost metrics:** runtime per stage, peak memory.
- Report per-fold values plus mean ± std; a gain smaller than the fold-to-fold std is not a real gain.

**Keep/reject rule (default, adjust if fold noise is larger):** keep a change only if mean macro F0.5 on folds 0–2 improves by more than ~3× the fold std (and normally ≥ 0.0005) **and** it does not lose on 2 of 3 folds, **and** runtime/memory stay within limits. For blocking changes, keep if ceiling rises materially (≥ ~0.3 pp) without more than ~1.5× candidates/S1.

---

## 3. Phase 0 — Reproduce Baseline and Diagnose Misses

Do this before any improvement.

1. **Do not retrain the baseline** (it costs ~2 hrs). Load the existing baseline outputs/models/predictions, recompute metrics on folds 0–2 from them, confirm the numbers above, and record them as experiment `E000`. Only re-run a stage if its cached output is missing or inconsistent.
2. Build a cache so blocking, features, and model stages can be re-run independently.
3. **Blocking miss analysis** — for every true pair not in the candidate set, record *why* it was lost:
   - not in top-k (rank of the true S1 if k were unlimited)
   - removed by the 0.4 relative cutoff
   - only shared tokens were above the DF cap
   - zero token overlap at all
   - country mismatch/missing country
4. Slice misses by: ASCII vs non-ASCII, country, missing name/address, exact/near-exact name, weak-name/strong-address, source (S2 vs S3).
5. **Classifier miss analysis** — among true pairs that *were* candidates but mispredicted: low probability, lost on margin, lost to a competing S1, below threshold.

Output a short table of miss causes. This decides which of Phases 1–4 matter most; do not skip it.

---

## 4. Improvement Phases (priority order)

Each phase is its own set of ablations. When the candidate set changes, features and model must be regenerated/retrained on the new candidates before scoring end-to-end.

### Phase 1 — Multi-Retriever Blocking
Add complementary retrievers and **union** candidates before final pruning:
- Character n-gram TF-IDF (test 2–4, 3–5 grams) — name, and name+address
- Name-only retrieval
- Address-only retrieval (words + numbers)
- Phonetic/skeleton-only retrieval
- Exact normalized-token retrieval for highly informative fields

Compare: (A) current blocker, (B) each retriever alone, (C) union, (D) union + dedup, (E) union + final pruning. Keep the cheapest configuration that materially raises recall/ceiling. Targeted goal: push blocking recall from 0.9585 toward ~0.98.

### Phase 2 — Adaptive K
Replace fixed `k=10` with a per-query budget:
- clear queries → 10; ambiguous → 20; low-score/very ambiguous → 50

Signals: best retrieval score, best-vs-second gap, score distribution, number surviving the cutoff, query-token informativeness, script type. Also test adaptive DF-cap expansion for hard queries. Do **not** set K=50 globally.

### Phase 3 — DF Cap and Relative Cutoff
Grid over DF cap ∈ {2,500, 5,000, 7,500, 10,000, selective 20,000} × relative cutoff ∈ {0.2, 0.3, 0.4, 0.5, 0.6}, including interaction with K. Use the Phase 0 miss causes to check whether low-best-score queries are disproportionately hurt by the cutoff, and whether high-DF tokens help when combined with rarer tokens even if they shouldn't create pairs alone. Prefer conditional expansion over globally expensive settings.

### Phase 4 — Native-Script Retrieval
Dedicated evaluation slice for non-ASCII records (~0.90 recall). Compare original-script-only, transliteration-only, and combined retrieval using original-script char n-grams, transliterated tokens, skeletons, and numeric/address fragments. Offline processing only. Document the specific failure modes found.

### Phase 5 — S1-Level / Cluster Consistency
Use strong S1↔S2 and S1↔S3 links plus S2↔S3 similarity as extra evidence, e.g. S2 is weak to S1 directly but strongly similar to an S3 that strongly matches S1. Candidate features: strongest connected S2/S3 evidence, cluster similarity, shared normalized fragments, conflicting S1 assignments. Keep propagation conservative (high-confidence links only) and evaluate as a separate ablation.

### Phase 6 — LightGBM Tuning with Optuna
Run **after** blocking/candidate changes are settled, because the optimal hyperparameters depend on the candidate distribution.

**Stage A — model only.** Freeze blocking, candidates, features, assignment, threshold, margin.
- Sampler: Optuna TPE (seeded); pruner: Hyperband (or median) with LightGBM pruning callback.
- Objective per trial: train on each of folds 0–2's train split → predict validation candidates → apply current assignment + threshold + margin → entity-level macro F0.5 → return the mean across folds. Log F1-macro, precision, recall, singleton accuracy as trial user attributes.
- Search space:

| Parameter | Range |
|---|---|
| learning_rate | 0.01–0.15 (log) |
| num_leaves | 16–255 |
| min_data_in_leaf | 20–500 |
| max_depth | −1 or 4–16 |
| feature_fraction | 0.6–1.0 |
| bagging_fraction | 0.6–1.0 |
| bagging_freq | 0–10 |
| lambda_l1 | 1e-8–10 (log) |
| lambda_l2 | 1e-3–20 (log) |
| min_gain_to_split | 0–1 |

- Iterations: `n_estimators` ceiling 1,500–2,500 with early stopping on validation logloss; record the best iteration. Do not force 400 trees.
- Budget: start with 50–100 trials; persist the study to SQLite so it can resume. Enqueue the baseline params as trial 0.
- Save best params, best model, and the full trial log.

**Stage B — joint.** Using the best Stage A params as a starting point, jointly tune LightGBM params + threshold + margin on folds 0–2 (see Phase 9).

Compare both stages directly against the baseline params under the identical protocol.

### Phase 7 — More Training Data + Hard Negatives
Current training uses candidates from ~8% of S1. Plot performance vs training size (8% → 15% → 30% → …) within memory limits. Prefer realistic hard negatives over random ones: high blocking similarity but false, same/similar names, nearby addresses, ambiguous top-1/top-2, common business names. Ensure coverage of non-ASCII and missing-field cases.

### Phase 8 — Feature Engineering
First audit the 35 features (importance, near-duplicates, leakage). Then test additions one group at a time:
- char n-gram similarity; exact normalized-field matches
- rare-token overlap, token informativeness, common vs rare overlap
- script indicators; original-script vs transliterated similarity
- name/address length ratios
- numeric agreement: house number, number-set overlap, postal/PIN where present
- candidate rank, retrieval score, top-vs-second gap, S1 crowding
- source-pair and country interactions

**Missing ≠ mismatch:** encode missingness explicitly (NaN or indicator), never as zero similarity.

### Phase 9 — Threshold, Margin, Calibration
1. Keep threshold/margin fixed while measuring pure model gains.
2. Then retune on folds 0–2 with a finer grid, **maximizing macro F0.5** (not F1 — the F1-optimal t=0.40/m=0.1 is recorded only and must not be applied).
3. Optionally test calibration (isotonic/Platt fit on folds 0–2 out-of-fold predictions).
4. Test adaptive thresholds by regime (candidate rank, score gap, singleton risk, candidate count, country, script) — only on slices large enough not to overfit.

Report the effect on F1-macro, F0.5, precision, recall, singleton accuracy. Historical gain from this lever was ~+0.0012; expect it to be small.

### Phase 10 — Learning-to-Rank (experimental, last)
LightGBM LambdaMART with candidates grouped by query entity. Compare end-to-end F1-macro/F0.5 under the full matching policy. Replace the binary classifier only if the gain is clear.

---

## 5. Experiment Management

- Directory `experiments/` with one config file per experiment and a single `experiments/results.csv` (append-only).
- Each row: experiment ID, parent experiment, change description, config hash, git commit, seed, folds, blocking recall, ceiling, candidates/S1, total pairs, F1-macro (per fold + mean ± std), macro F0.5, precision, recall, singleton accuracy, runtime per stage, peak memory, model params, best iteration, threshold, margin, keep/reject decision.
- One change at a time: baseline → change → measure → keep/reject → next. After individually useful changes are found, test combinations (gains may overlap).
- Maintain `experiments/SUMMARY.md` with the running best configuration and why each kept change was kept.

---

## 6. Final Evaluation Protocol

1. Select the best configuration using folds 0–2 only.
2. Freeze preprocessing, blocking, features, model params, threshold, margin, assignment.
3. Evaluate **once** on folds 3–4. No changes after this point based on those numbers.
4. Report baseline vs final on: F1-macro, macro F0.5, precision, recall, singleton accuracy, blocking recall, ceiling, candidates/S1, runtime, peak memory.
5. Attribute the improvement to each kept change (ablation table).
6. For the submission run, retrain the frozen configuration on all labelled data (same params and best-iteration count), then run inference.
7. Verify `candidate_pairs.tsv` exactly matches the inference candidate set and `matching_results.tsv` ⊆ candidates.
8. Run the official submission validator before declaring done.
9. Save a reproducible checkpoint: config, model, seed, commit, and a one-command rerun script.

---

## 7. Key Principle

Blocking recall is 0.9585 and precision is already 0.987, so the gains are in **recovering true candidates**, not a more complex classifier. Diagnose misses first, fix retrieval second, tune the model third, and treat decision-rule tuning as a small finishing step. The deliverable is a measurable, reproducible improvement over the baseline, not a more complicated pipeline.