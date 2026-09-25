"""Shared paths, IO, logging and memory helpers for the entity-resolution pipeline."""
import os
import sys
import time

import psutil
import pyarrow as pa
import pyarrow.csv as pcsv

# Project layout: <root>/code/business_entity_resolution/src/common.py
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("ER_ROOT", os.path.dirname(os.path.dirname(PKG)))
DATA = os.environ.get("ER_DATA", os.path.join(ROOT, "student_resource", "dataset"))
WORK = os.environ.get("ER_WORK", os.path.join(ROOT, "work"))
OUT = os.environ.get("ER_OUT", os.path.join(ROOT, "output"))
LOGS = os.path.join(WORK, "logs")

# Refuse to start a heavy step when free system RAM is below this (GB).
MIN_FREE_GB = float(os.environ.get("ER_MIN_FREE_GB", "1.5"))

_T0 = time.time()
_PEAK = 0


def wpath(*parts):
    """Path inside the work dir, creating parent folders."""
    p = os.path.join(WORK, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def rss_gb():
    return psutil.Process().memory_info().rss / 2**30


def free_gb():
    return psutil.virtual_memory().available / 2**30


def log(*msg):
    """Print and append to work/logs/run.log with elapsed time, RSS and free RAM."""
    global _PEAK
    r = rss_gb()
    _PEAK = max(_PEAK, r)
    line = f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:7.1f}s rss={r:5.2f}G peak={_PEAK:5.2f}G free={free_gb():5.2f}G] " + " ".join(map(str, msg))
    print(line, flush=True)
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "run.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def guard(what=""):
    """Abort cleanly (checkpoints stay on disk) instead of letting the OS thrash."""
    if free_gb() < MIN_FREE_GB:
        log(f"ABORT {what}: free RAM {free_gb():.2f}G < {MIN_FREE_GB}G; rerun resumes from last checkpoint")
        sys.exit(3)


def read_tsv(path):
    """Read an official TSV verbatim (no quote handling, empty strings kept as '')."""
    return pcsv.read_csv(
        path,
        parse_options=pcsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pcsv.ConvertOptions(strings_can_be_null=False, column_types={
            "entity_id": pa.string(), "business_name": pa.string(), "business_address": pa.string(), "country": pa.string(),
            "source1_entity_id": pa.string(), "matched_entity_ids": pa.string()}),
    )


def src_file(split, src):
    return os.path.join(DATA, split, f"{split}_source{src}.tsv")


def id_str(src, num):
    """Rebuild the exact original entity_id from (source, integer part)."""
    return f"S{src}-{num}"
