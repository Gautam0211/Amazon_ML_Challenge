# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** TEAM_NAME_TBD
**Team Members:** TBD
**Submission Date:** 2026-09-25

---

## 1. Executive Summary

The pipeline has three stages:

1. **Blocking:** an IDF-weighted sparse cosine over hashed name, phonetic-skeleton and address tokens retrieves the top S1 candidates for every S2/S3 record, within the same country label.
2. **Scoring:** a LightGBM classifier scores each candidate pair on 35 string, number, missingness and blocking-context features.
3. **Assignment:** each S2/S3 record goes to its single best-scoring S1, and only when the probability clears a threshold tuned directly for the official per-S1 macro-F0.5.

Everything is offline, rule-based or LightGBM (MIT licence), CPU-only, and fits in about 4 GB RAM. No rule depends on a specific country, which matters because France appears only in test.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from the training data (observed, not guaranteed):

- **Size:** 2.21M S1, 5.03M S2 and 5.29M S3 records, with 7.64M true pairs. S1 has 3.46 matches on average (median 3, max 11), and **5.6% of S1 are singletons**.
- **Ownership:** no S2/S3 record is linked to more than one S1 (0 of 7.6M), and ~26% of S2/S3 records match nothing.
- **Country:** the label is identical on 100% of true pairs. Test adds **France** (15% of test S1), which has no labels.
- **Name noise:** only 10.7% of true pairs have equal names ignoring case. Variations include typos, word swaps, legal-suffix changes (Pvt Ltd ↔ Private Limited), DBA/trade names, websites and handles (`prudencegenesis.com`, `@tennissportive`), and completely unrelated fake names at the same address.
- **Scripts:** 15% of S2 names and 11% of S3 names are in native Indian scripts (Devanagari, Kannada, Tamil, Telugu). Some state names are too.
- **Address noise:** abbreviations, reordered components, dropped city/state/PIN, changed or truncated house numbers (6885 → 688), `<NULL>`/`N/A` placeholders, and 3.4% of S2/S3 addresses empty.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier + constrained Assignment.
**Core Innovation:**
- **Retrieval from the S2/S3 side.** Each S2/S3 record retrieves its own top S1 candidates, which mirrors the "each S2/S3 belongs to at most one S1" structure.
- **Competition features.** The classifier sees how the pair compares with the S2/S3 record's *other* candidates, and these are the strongest features.
- **Metric-level tuning.** The assignment threshold is tuned on the exact official metric, singletons included.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** each record becomes a set of 64-bit hashed tokens, all namespaced by the record's country label:
  - normalized core-name words (legal words removed)
  - the concatenated core name, which matches website/handle names
  - phonetic consonant skeletons of name words, which bridge transliteration and vowel typos
  - normalized address words and numbers
- **Scoring:** IDF-weighted cosine similarity, computed as sparse matrix products (S2/S3 chunk × S1ᵀ) separately per country. Only tokens whose S1 document frequency is ≤ 5,000 generate pairs; common tokens still count in the norm. Candidates below 40% of a query's best score are dropped, and the **top 10 S1 per S2/S3 record** are kept.
- **Candidate pairs generated:** v1 (cap 500): 93.5M train pairs (42 per S1) and 90.8M test pairs (52 per S1); reduction ratio 0.999996. **v2 (cap 5,000, rel 0.4):** 62.5M train pairs (28.3 per S1), reduction ratio 0.9999973; 60.3M test pairs (34.8 per S1). Train Blocking took 49 min with a 3.5 GB peak.
- **How true matches were kept:** recall was measured on the ground truth for every setting (table in §5). A 2% query sample showed the frequency cap was the main recall loss, so it was raised from 500 to 5,000 (recall@10: 0.869 → 0.958 on the sample). Chunk sizes adapt to free RAM, so blocks cannot explode memory.

---

## 4. Matching Model

**Features used (35):**
- **Name features:** RapidFuzz ratio, token-set, token-sort and partial ratio on the full normalized name and on the core name; Jaro-Winkler on the core name; space-free ratio (website names); token Jaccard; phonetic-skeleton token-set similarity and Jaccard; token counts.
- **Address features:** ratio, token-set, token-sort and partial ratio; token Jaccard; Jaccard of all numbers; first-number (house number) equality; number counts on each side.
- **Other:** blocking cosine; blocking rank; gap to the best S1 and to the runner-up S1 for the same S2/S3 record; S1-side crowding (number of candidates, how often this S1 ranks first, gap to the S1's best score); source (S2/S3); native-script flag; empty-address flag. Missing values are encoded as −1 (“unknown”), separate from a real mismatch.

**Model type:** LightGBM binary classifier (400 trees, 127 leaves, learning rate 0.08). It is trained on all candidate pairs of a random 8% of S1 entities (7.36M pairs, 7.2% positive), so the negatives are realistic hard negatives from Blocking. Validation uses 5-fold out-of-fold predictions grouped by S1, so no S1 is scored by a model that saw it.

**Threshold selection method:** the Assignment policy is tuned on folds 0–2 with the exact per-S1 macro-F0.5 (singletons score 1 only for an empty prediction). Each S2/S3 record is assigned to its argmax-probability S1 if p ≥ t and p − p_second ≥ margin. The chosen policy was t = 0.63 with margin 0.2 for v1, and **t = 0.60 with margin 0.2 for v2**. The policy was then frozen and reported once on untouched folds 3–4.

---

## 5. Results & Error Analysis

| Version | Blocking recall (train) | Holdout macro-F0.5 | Pair precision | Pair recall | Singleton acc. |
| --- | --- | --- | --- | --- | --- |
| v1 (cap 500) | 0.869 | **0.9147** | 0.9885 | 0.8344 | 0.960 |
| **v2 (cap 5000, final)** | **0.9585** | **0.9602** | 0.9870 | 0.9213 | 0.958 |

Blocking ceiling (a perfect classifier on the candidates): 0.938 for v1 and **0.985 for v2**. In v2 the remaining loss is split about evenly: ~0.025 from classification/Assignment and ~0.015 from Blocking. In v1, 13.1% of the true holdout pairs never became candidates and 3.5% were rejected by the model.

Assignment ablation (v1, tune folds): threshold-only 0.9134 vs Assignment 0.9143 (both at their best threshold). The margin added +0.0002.

Negative experiment: merging voiced/unvoiced consonants in the skeleton (b/p, d/t, g/k) lowered non-ASCII recall (0.9020 → 0.8977), so it was reverted.

- **Common false positives (wrong merges):** the same or a similar name at a *different* house or unit number on the same street, and sibling companies that differ by one word ("… Group", "… Ventures").
- **Common false negatives (missed matches):** records whose name was replaced by an unrelated trade name; native-script names with partial addresses; changed city names; one side with an empty address and a generic name.

---

## 6. Conclusion

The metric forgives missing links more than wrong ones, so the design targets precision: country-consistent blocking, a classifier that sees the competing candidates, and a one-S1-per-record assignment tuned on the official metric. Candidate recall was the dominant limit in v1. Raising the blocking frequency cap from 500 to 5,000 lifted train recall@10 from 0.869 to 0.9585 and holdout macro-F0.5 from 0.9147 to **0.9602**, the largest single gain. France is handled without labels through country-agnostic rules, and its output statistics resemble the labelled countries, but France accuracy could not be measured.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/` (entry points run in this order; the README lists exact commands):

| File | Stage |
| --- | --- |
| `prep.py`, `prep_worker.py`, `normalize.py` | Normalization + Transliteration (anyascii), phonetic skeletons, hashed blocking tokens |
| `labels.py` | ground truth → integer pairs, S1-grouped folds |
| `block.py`, `eval_block.py`, `diag_recall.py` | Blocking and recall diagnostics |
| `features.py` | Feature Engineering (batched, resumable part files) |
| `model.py` | LightGBM OOF, Assignment policy search, Validation, final model |
| `metric.py` | official per-S1 macro-F0.5 (self-test reproduces the statement example, 0.714) |
| `predict.py` | test inference → `output/matching_results.tsv`, `output/candidate_pairs.tsv` |
| `package.py` | builds the submission zip |
| `common.py` | paths, logging, memory guard |

### B. Additional Results

Blocking settings on a 2% query sample (train):

| cap | rel | recall@1 | recall@10 | pairs/query | time |
| --- | --- | --- | --- | --- | --- |
| 500 (v1, full train) | 0.25 | 0.792 | 0.869 | 9.1 | 14.8 min full |
| 2000 | 0.25 | 0.877 | 0.932 | 17.4 | 62 s |
| 5000 | 0.4 | 0.912 | 0.958 | 10.0 | 78 s |
| 5000 | 0 (k 50) | 0.912 | 0.958 | 50 | 549 s |
| 20000 | 0.4 | 0.934 | 0.972 | 9.4 | 316 s |
