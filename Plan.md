# Amazon ML Challenge 2026 — Business Entity Resolution
## Claude Code: End-to-End Project Resumption and Execution Plan

> **Role:** Lead ML engineer  
> **Operating mode:** Execute autonomously in the existing local project; preserve completed work.  
> **Primary objective:** Maximize the competition's **per-Source-1 macro-averaged F0.5** score while maintaining strong candidate recall, memory-safe execution, reproducibility, and generalization to the unseen country, France.  
> **Submission control:** Prepare and validate every deliverable, but **do not upload to the competition portal without my explicit approval**.

---

## 1. First Action: Audit the Existing Project

Before writing new code or rerunning experiments, inspect the entire local project:

- Official challenge guidelines and problem statement.
- Training and test data, ground truth, and dataset structure.
- Existing scripts, installed dependencies, notebooks, configuration, and generated files.
- Profiling reports, logs, errors, checkpoints, previous experiments, and available outputs.

**Verify actual progress rather than trusting the previous status report.** Preserve all usable code, datasets, successful experiments, and outputs. Resume from the last valid checkpoint rather than rebuilding completed work.

### Previously reported status — verify before proceeding

- Dataset extracted; tools and dependencies installed; initial profiling performed.
- The latest profiling script reportedly crashed with a `MemoryError`. Long string IDs such as `S2-166376419` were identified as a possible source of excess memory use; inspect the failure before assigning a definitive cause.
- No model has reportedly been trained or submitted.

## 2. Execution Rules and Success Criteria

Build a **reproducible, memory-efficient, end-to-end entity resolution pipeline** that:

1. Generates plausible Source 1 (S1) ↔ Source 2/3 (S2/S3) candidate pairs.
2. Scores candidate pairs using meaningful similarity features and a LightGBM baseline.
3. Applies an assignment policy and threshold tuned to the **official macro-F0.5** metric.
4. Produces complete, correctly formatted test outputs and passes the official validator.
5. Can be reproduced from the packaged code and provided challenge data.
6. Processes unseen country labels, including **France**, without hard-coded US/India assumptions.

**Prioritize a working, measured baseline before experimentation.** Never invent experimental results, scores, timings, or completed work.

### Non-negotiable local memory safeguards

Inspect available RAM, disk space, and the size of the input files before heavy jobs. Run all experiments **locally** with memory-safe processing:

- Replace verbose internal string IDs with **compact numeric, source-aware identifiers** only after proving a **lossless, reversible mapping** back to every exact original `S1-`, `S2-`, and `S3-` ID; never lose source prefixes or leading zeros on output.
- Read and process datasets **in chunks** where useful. Use efficient column types, sparse representations when appropriate, and disk-backed intermediate artifacts.
- Do **not** materialize the full S1 × (S2 + S3) Cartesian product or all candidates in RAM.
- Bound block sizes and per-stage working sets; monitor peak RAM and adapt batch sizes downward before memory pressure becomes critical.
- Save deterministic intermediate checkpoints. On failure, preserve logs and resume from the last completed stage rather than repeating the entire run.
- Do not treat numeric IDs alone as a complete fix for the reported `MemoryError`: identify and address the actual high-memory operation.

---

## 3. Verify the Data Assumptions

Reproduce and record the initial profiling statistics. Distinguish observed training-data patterns from rules guaranteed by the problem statement.

Verify:

- Duplicate or malformed IDs and records.
- Singleton prevalence and true-match count per S1.
- Whether any S2/S3 record maps to multiple S1 records in the provided ground truth.
- Country consistency between true twins.
- Missing fields, different writing scripts, noisy names, address patterns, and data-quality issues.
- Training versus test distributions, especially the appearance of **France only in test**.

Do **not** hard-code country names, source counts, singleton percentages, distractor percentages, or assumptions about one-to-one ownership unless the official specification or verified data supports them.

---

## 4. Implement and Measure the Eight-Stage Pipeline

### Stage 1 — Normalization

Clean business names and addresses using context-aware rules for case, punctuation, Unicode, abbreviations, legal suffixes, and missing fields. Preserve raw input and meaningful identifiers/tokens. Avoid transformations that make distinct businesses look identical.

**Measure:** Coverage, representative before/after examples, runtime, and memory usage.

### Stage 2 — Transliteration

Add **offline, reproducible transliteration** for relevant Indian scripts and normalization appropriate for accented Latin text, including French names. Keep original-script versions alongside transformed representations when useful. Measure whether transliteration improves validation performance rather than assuming it does.

**Measure:** Candidate recovery attributable to transliteration and its effect on the end-to-end score.

### Stage 3 — Blocking (Candidate Generation)

Create complementary ways to find plausible pairs using rare name tokens, character n-grams, phonetic similarity, informative address components, and other locally derived clues. Cap overly common tokens and oversized blocks so they cannot explode RAM usage.

Blocking determines the **maximum recall** achievable downstream: a true twin not generated as a candidate cannot be recovered by LightGBM.

**Measure:** Candidate recall on held-out ground truth, candidates per S1, reduction ratio, runtime, and peak RAM. Improve Blocking when it misses too many genuine twins.

**Output integrity:** `candidate_pairs.tsv` must represent the **final candidate set actually passed to the inference model**, after any later candidate filtering—not an intermediate, unfiltered block.

### Stage 4 — Feature Engineering

For each surviving candidate pair, compute useful similarity features such as:

- Name similarity: Levenshtein/edit distance, Jaccard, TF-IDF cosine, character n-gram and phonetic measures.
- Address similarity: token and string similarity, house number, postal code, and other informative address components.
- Cross-field agreement, contradictions, and country consistency where applicable.
- Explicit missingness indicators to distinguish **unknown** from **mismatch**.

Generate features in memory-safe batches. Inspect feature quality and avoid target leakage.

**Measure:** Feature coverage, importance, runtime, and peak RAM.

#### Feature Selection and Optimization

**Principle:** Use multiple complementary similarity signals and let LightGBM learn their combined value. Retain features on the basis of measured end-to-end improvement, not popularity or feature importance alone.

1. **Establish a lightweight baseline** with normalized exact matches, token similarity, edit distance, address components, missingness indicators, country equality, source, and Blocking evidence.
2. **Test additional features**, including Jaro-Winkler and character TF-IDF cosine, using controlled experiments on the same validation splits.
3. **Run feature ablation:** remove one feature or related feature group at a time, retrain, and measure the change in **per-S1 macro-F0.5**.
4. **Measure efficiency:** record runtime and peak RAM for each configuration. Compute inexpensive features first; calculate more expensive similarities in memory-safe batches, without discarding plausible candidate pairs solely to save memory.
5. **Select and document** the final feature combination using validation performance, consistency across folds, memory consumption, and runtime. Record retained and removed features and the evidence behind each decision in `PROGRESS.md`.

### Stage 5 — Model (LightGBM)

Train a strong, reproducible **LightGBM baseline** using labeled positive pairs and appropriately sampled negatives, including **hard negatives**—plausible lookalikes from Blocking that are actually different businesses.

Check class imbalance, leakage, training/inference consistency, and whether probability calibration would improve assignment. Prefer measurable gains over unnecessary model complexity.

**Measure:** Validation diagnostics, resource use, and end-to-end contribution to macro-F0.5.

### Stage 6 — Assignment and Threshold

Score candidates, then decide which matches to retain. Evaluate per-pair thresholds and, **only if the observed ground truth supports the constraint**, assign each S2/S3 record to its highest-scoring S1 or leave it unmatched.

Tune the **complete decision policy** on held-out validation predictions against the official **per-S1 macro-F0.5**, not ordinary global/pairwise F0.5. Pay special attention to false merges, singletons, and ambiguous near-duplicates.

**Measure:** Macro-F0.5, precision, recall, singleton accuracy, and false merges.

#### Threshold Tuning and Assignment Optimization

**Objective:** Optimize the *complete matching pipeline* for the official **per-S1 macro-F0.5**, including correctly identified singletons. Tune thresholds **after** applying the candidate-to-S1 Assignment policy, rather than optimizing conventional pairwise F0.5 or ROC-AUC in isolation.

1. **Obtain leakage-safe predictions.** Generate held-out or out-of-fold LightGBM scores using entity-grouped validation. Keep the relevant competing S1 candidates together within each validation evaluation so Assignment is tested realistically.
2. **Apply Assignment.** If verified ground truth supports it, let each S2/S3 record belong to its highest-scoring S1 or to nobody. Define deterministic tie-breaking and preserve all S1 rows, including singletons.
3. **Search thresholds across the full score range (0.00–1.00).** Start with a coarse sweep, including values below 0.50, then refine near promising regions; where practical, evaluate thresholds at observed prediction-score boundaries. Do not assume a threshold of 0.50 is optimal or that LightGBM scores are calibrated probabilities.
4. **Score every configuration end to end.** For each threshold, apply Assignment and compute the **exact official per-S1 macro-F0.5**. Include singleton scoring; also record pairwise diagnostic precision and recall, false merges, and singleton accuracy.
5. **Compare decision policies.** Use one global threshold as the baseline. Then test S2/S3-specific thresholds and a confidence margin between the best and second-best competing S1 scores. Introduce extra parameters only when validation demonstrates a meaningful, repeatable benefit.
6. **Check robustness and prevent overfitting.** Compare configurations across independent validation folds. Where resources allow, select thresholds on tuning folds and report the selected policy on a separate untouched holdout. Prefer the simpler, stable configuration over marginal gains from a complex rule.
7. **Freeze and deploy.** After policy selection, retrain the chosen model on the full training set, freeze its Assignment rules and thresholds, and run test inference **without using leaderboard feedback to tune the policy**.
8. **Maintain an audit trail.** Record every tested threshold/configuration, actual measured scores, selected policy, runtime, resource use, and decision rationale in `PROGRESS.md`.

### Stage 7 — Validation

Create a leakage-safe, entity-grouped holdout from training data. Score each S1 separately and average those scores according to the official formula. For a singleton, a correctly predicted empty list earns **1.0**; any predicted match earns **0.0**.

Evaluate Blocking and the full pipeline separately so the main bottleneck is visible. Include reasonable cross-country/generalization stress tests, but **never claim to possess labeled France validation data** when France is absent from the supplied training labels.

**Report:** Official macro-F0.5, pairwise diagnostic precision/recall, Blocking recall, singleton accuracy, false merges, runtime, peak RAM, and limitations.

### Stage 8 — Output and Official Validation

Write these two tab-separated files for **every S1 in the test set**, using the exact original IDs and official column names:

- `output/matching_results.tsv`: `source1_entity_id` and `matched_entity_ids`.
- `output/candidate_pairs.tsv`: `source1_entity_id` and `candidate_entity_ids`.

Produce exactly one row per test S1; use empty ID lists for no matches/candidates. Allow only valid test S2/S3 IDs, with no duplicates. Every final match must also occur in the corresponding final candidate set.

Run the supplied validator and resolve all issues until it reports **PASS**:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The official validator checks file validity, **not model accuracy**. Record the actual held-out macro-F0.5 separately.

---

## 5. Experiment and Improvement Loop

1. Get the simplest complete, **measured baseline** working as early as possible. Preserve its code, parameters, outputs, and metrics.
2. Diagnose the limiting factor: candidate recall, model discrimination, assignment decisions, validation generalization, runtime, or memory.
3. Run focused experiments that change a controlled component; compare results on a consistent validation split.
4. Retain improvements supported by actual metrics and reproducible runs. Log unsuccessful experiments and why they failed.
5. Stop excessive experimentation when it jeopardizes the challenge deadline or completion of a valid final package.

Never present estimates as measured results. Do not let public leaderboard feedback substitute for a leakage-safe local validation process.

## 6. Competition Compliance

- Resolve identities **only from the provided challenge datasets**.
- **Do not** use external business databases, government registries, commercial entity-resolution services, geocoding APIs, internet-based business lookups, or external data augmentation.
- **Do not** transmit challenge business records to external AI services.
- Check that the final model satisfies the challenge's **MIT/Apache 2.0 license** requirement and **maximum 8-billion-parameter** restriction.
- Preserve the exact official file schemas, maintain submission version history, and do not spend a leaderboard submission slot without my explicit approval.
- Keep the local project and final submission reproducible and auditable.

---

## 7. Required Deliverables

Build and verify the official final package structure:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/                 # Complete, commented source code
│       ├── README.md            # Exact data → blocking → matching → output commands
│       └── requirements.txt     # Pinned dependencies
└── Documentation_template.md   # Completed methodology and experimental report
```

Also prepare the requested **1–2-page ML approach summary**, covering the approach, models, experiments, and conclusions. Explain the methodology, Blocking strategy, model architecture, Feature Engineering, actual experimental results, and unresolved limitations. Use the official documentation template for the packaged methodology report.

## 8. Plain-Language Progress Report

Maintain `PROGRESS.md` **throughout execution**, not only at the end. Write it for a non-technical project owner: explain each stage simply, **retaining the original terminology**—Normalization, Transliteration, Blocking, Feature Engineering, LightGBM, Assignment, Validation, and Output.

For every stage, answer:

1. **What is it?** Explain the technical term in very simple language.
2. **Why do we need it?** Explain its role in finding real twins and avoiding false matches.
3. **What did you run?** Include exact commands or reproducible references.
4. **What happened?** Show actual measured results, not guesses.
5. **What broke, and how was it fixed?** Include enough detail for someone else to debug it.
6. **What happens next?** Distinguish completed, in-progress, and pending work.

Include a specific explanation of the reported `MemoryError`, its verified root cause, and how the fix was tested. Keep a visible summary of the **best measured macro-F0.5**, **Blocking candidate recall**, **peak RAM**, and **official validator status**.

---

## 9. Execution Autonomy and Final Handoff

Proceed through the stages without unnecessary approval pauses. Ask only when an essential resource is unavailable, an irreversible action requires approval, or a competition submission would occur. **Do not upload any file to the challenge portal without my approval.**

At completion, provide a concise executive summary identifying:

- The stages actually completed and the baseline/improvements actually tested.
- The exact output, source-code, documentation, and ZIP paths.
- Best **measured** macro-F0.5 and candidate recall, plus runtime and peak RAM.
- Official validator result (**PASS**, or the actual failure and remaining blocker).
- Unresolved risks, particularly memory behavior and generalization to France.
- The exact next action required from me before any competition submission.

**Definition of done:** A reproducible local pipeline has produced both official TSVs; the official validator reports PASS; the documentation and source package are complete; the submission ZIP is prepared; and no leaderboard submission has occurred without explicit approval.
