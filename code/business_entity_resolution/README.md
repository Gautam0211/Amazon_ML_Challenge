# Business Entity Resolution — reproducible pipeline

Links each Source 1 business to its Source 2 / Source 3 records. CPU only, about 16 GB RAM, no external data and no network calls.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
```

Tested with Python 3.14.5 on Windows 11 (16 threads, 15.4 GB RAM).

## Data layout

By default the code expects this layout (you can override the paths with the `ER_ROOT`, `ER_DATA`, `ER_WORK` and `ER_OUT` environment variables):

```
<root>/student_resource/dataset/{train,test}/*.tsv   # official data
<root>/code/business_entity_resolution/src/          # this code
<root>/work/                                         # intermediate Parquet files (created)
<root>/output/                                       # final TSVs (created)
```

## Run end to end (from `code/business_entity_resolution/src`)

```bash
python prep.py --split train        # Normalization + Transliteration  -> work/train/s{1,2,3}.parquet
python prep.py --split test
python labels.py                    # ground truth -> integer pairs, 5 S1-grouped folds
python block.py --split train       # Blocking -> work/train/cand.parquet
python eval_block.py                # candidate recall report
python features.py --split train    # Feature Engineering -> work/train/feat/part_*.parquet
python model.py                     # LightGBM OOF + Assignment policy search + Validation + final model
python block.py --split test
OPENBLAS_NUM_THREADS=1 python features.py --split test --workers 3 --batch 150000   # lower-memory settings used for the final run
python predict.py                   # -> output/matching_results.tsv, output/candidate_pairs.tsv
cd ../../../student_resource && python utils/validate_submission.py \
    --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
```

Every step checks for its outputs and skips work that is already finished. After a crash or a memory abort (exit code 3), rerun the same command and it resumes.

## Memory safety

- `common.guard()` stops a step cleanly when free RAM falls below `ER_MIN_FREE_GB` (default 1.5).
- Blocking chunk sizes shrink automatically when free RAM drops below 3 GB.
- If `features.py` fails with a `MemoryError` or an OpenBLAS allocation error, rerun it with `OPENBLAS_NUM_THREADS=1` and fewer `--workers` or a smaller `--batch`. Finished parts are kept.
- Records are identified internally by (source, integer ID part). `prep.py` asserts that `"S<src>-" + str(int)` rebuilds every original ID exactly.

## Licences

The only model is LightGBM (MIT). Libraries: numpy, scipy, pyarrow, rapidfuzz, psutil and lightgbm are BSD, Apache or MIT; anyascii is ISC. There are no pretrained weights and no LLM.
