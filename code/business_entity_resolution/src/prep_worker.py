"""Worker for prep.py. Kept free of pandas/pyarrow so each spawned process stays small (Windows commit limit)."""
from hashlib import blake2b

import numpy as np

from normalize import normalize_record


def tok_hash(s):
    """Deterministic signed 64-bit token hash (same value in every process and run)."""
    return int.from_bytes(blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


def block_tokens(country, core, addr, skel):
    """Blocking tokens for one record. Namespaced by country, so blocks never cross countries."""
    c = country.strip().lower()
    toks = {f"{c}\x1fn\x1f{t}" for t in core.split() if len(t) >= 2 and not t.isdigit()}
    parts = [t for t in core.split() if not t.isdigit()]
    if len(parts) >= 2:  # 'prudence genesis' also indexed as 'prudencegenesis' (matches website names)
        toks.add(f"{c}\x1fn\x1f{''.join(parts)}")
    toks.update(f"{c}\x1fk\x1f{t}" for t in skel.split())
    toks.update(f"{c}\x1fa\x1f{t}" for t in addr.split() if len(t) >= 2 or t.isdigit())
    return toks


def work(args):
    """Normalize one chunk: returns normalized columns, per-record token counts and flat token hashes."""
    names, addrs, countries = args
    out = {k: [] for k in ("name_norm", "name_core", "addr_norm", "name_skel")}
    lens, flat = [], []
    for n, a, c in zip(names, addrs, countries):
        nn, core, an, sk = normalize_record(n, a)
        out["name_norm"].append(nn); out["name_core"].append(core); out["addr_norm"].append(an); out["name_skel"].append(sk)
        t = block_tokens(c, core, an, sk)
        lens.append(len(t)); flat.extend(tok_hash(x) for x in t)
    out["nonascii"] = [not x.isascii() for x in names]
    return out, np.asarray(lens, np.int32), np.asarray(flat, np.int64)
