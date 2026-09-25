# ML Approach Summary: Business Entity Resolution (Amazon ML Challenge 2026)

## Problem
For each Source 1 (S1) business, find every Source 2/Source 3 record that describes the same business. The score is the per-S1 macro-averaged F0.5, and a singleton S1 counts as correct only when its prediction is empty. Train has 2.21M S1, 5.03M S2 and 5.29M S3 records with 7.64M true pairs (3.46 per S1, 5.6% singletons). Test has 1.73M S1 and adds **France**, which has no training labels. Names are very noisy (only 10.7% of true pairs share a name ignoring case), 11–15% of S2/S3 names are in Indian native scripts, and ~3.4% of addresses are empty.

## Pipeline (CPU only, about 4 GB peak RAM)
1. **Normalization and Transliteration.** Unicode NFKC, lower-casing and punctuation cleanup; legal-suffix and abbreviation normalisation (Pvt Ltd ↔ Private Limited, St ↔ Street); `<NULL>`/`N/A` treated as missing; offline `anyascii` transliteration of Devanagari/Tamil/Telugu/Kannada and of accented Latin (French); vowel-free phonetic skeletons. Records get compact integer IDs, and `prep.py` asserts that each one maps back to its exact original ID.
2. **Blocking.** Every S2/S3 record retrieves its own top 10 S1 candidates, within the same country label, by IDF-weighted sparse cosine over hashed tokens:
   - core-name words
   - the concatenated core name (for websites and handles)
   - phonetic skeletons
   - address words and numbers

   Tokens with S1 document frequency above 5,000 do not generate pairs, and candidates below 40% of the query's best score are dropped. Chunk sizes adapt to free RAM.
3. **Feature Engineering (35 features).** On name, core name and address: RapidFuzz ratio, token-set, token-sort and partial ratio, plus Jaro-Winkler, token Jaccard, skeleton similarity, number/house-number agreement and counts. Blocking-context features compare the pair with the record's other candidates (rank, gap to best and runner-up, S1 crowding). Missing values are encoded separately from mismatches.
4. **LightGBM.** A binary classifier (400 trees, 127 leaves), MIT licence, far below 8B parameters. It trains on all Blocking candidates of an 8% S1 sample (7.2% positives), so the negatives are realistic hard negatives. 5-fold out-of-fold scoring is grouped by S1.
5. **Assignment.** Each S2/S3 record goes to its single best-scoring S1 only if p ≥ t and p − p₂ ≥ margin. The training data shows that no S2/S3 belongs to two S1s. The policy is tuned on the official macro-F0.5 on folds 0–2 and then frozen.
6. **Validation and Output.** The policy is reported once on untouched folds 3–4. `candidate_pairs.tsv` is exactly the set scored by the model, so every predicted match is inside it. The official validator (with `--check-ids`) reports PASS.

## Results (measured on the train holdout, folds 3–4)

| Version | Blocking recall@10 | Ceiling* | macro-F0.5 | Precision | Recall | Singleton acc. |
| --- | --- | --- | --- | --- | --- | --- |
| v1 (df cap 500) | 0.869 | 0.938 | 0.9147 | 0.9885 | 0.834 | 0.960 |
| **v2 (df cap 5000, final)** | **0.9585** | 0.985 | **0.9602** | 0.9870 | 0.921 | 0.958 |

\*Ceiling is the score of a perfect classifier on the Blocking candidates.

## Key experiments
- **Where the v1 loss came from:** 13.1% of true pairs never became candidates, against 3.5% rejected by the model, so Blocking was the weakest stage.
- **Blocking frequency cap:** raising it from 500 to 5,000 was the biggest win (+0.046 macro-F0.5). A cap of 20,000 gives recall@10 of 0.972 but was 4× slower, so it wasn't adopted.
- **Voiced/unvoiced consonant merging in skeletons:** it lowered non-ASCII recall, so it was reverted.
- **Assignment vs threshold only:** Assignment added +0.001, and the margin added +0.0002.

## Engineering notes
The initial `MemoryError` came from holding millions of string IDs and full string columns in Python objects. The fix was integer IDs with a verified lossless mapping, Parquet part files, adaptive chunking, a free-RAM guard, and resumable batched stages. The machine's real limit was about 5 GB of free commit memory, not physical RAM. In the final test run the feature step first failed on OpenBLAS allocations. Rerunning it with single-threaded BLAS, 3 workers and 150k-row batches completed it in about 20 min.

## Limitations and next steps
- France accuracy cannot be measured because it has no labels. The design is country-agnostic, and France output statistics look similar to the labelled countries.
- The native-script path is still the weakest: non-ASCII recall is about 0.90, against about 0.96 overall.
- Remaining headroom is roughly 0.025 in classification/Assignment and 0.015 in Blocking. Possible next steps are more training data (above 8% of S1), an S1-level cluster-consistency step, and a higher frequency cap made affordable by parallel blocking.
- **Compliance:** only the provided data was used. There were no external lookups, APIs or LLMs; the only model is LightGBM (MIT).
