"""Memory-lean run of the official validator for large candidate files.

The official validator keeps every candidate ID as a Python string (~10 GB for 93M IDs).
This runs its own validate_id_list_file on the matching file whole and on the candidate
file in row chunks, then applies the file-level rules (S1 coverage, duplicate rows) and the
matched-subset-of-candidate soft check across chunks.

Usage: python src/validate_lean.py --out-dir output --test-dir student_resource/dataset/test [--check-ids]
"""
import argparse
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "student_resource", "utils"))
import validate_submission as vs  # noqa: E402


def chunks(path, rows):
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        buf = []
        for line in f:
            buf.append(line)
            if len(buf) == rows:
                yield header, buf
                buf = []
        if buf:
            yield header, buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--test-dir", default="student_resource/dataset/test")
    ap.add_argument("--check-ids", action="store_true")
    ap.add_argument("--rows", type=int, default=200_000)
    a = ap.parse_args()
    errors, warnings = [], []
    required = vs.read_ids(os.path.join(a.test_dir, "test_source1.tsv"))
    valid = vs.load_match_targets(a.test_dir, warnings) if a.check_ids else None
    print(f"  required S1 entities: {len(required)}  check_ids={valid is not None}")

    matched = vs.validate_id_list_file(os.path.join(a.out_dir, "matching_results.tsv"), vs.MATCHING_HEADER,
                                       "matched_entity_ids", required, valid, errors)

    seen, dups, offenders, n = set(), set(), set(), 0
    tmp = os.path.join(tempfile.gettempdir(), "cand_chunk.tsv")
    for header, buf in chunks(os.path.join(a.out_dir, "candidate_pairs.tsv"), a.rows):
        ids = {l.partition(vs.DELIM)[0] for l in buf}
        dups |= ids & seen
        seen |= ids
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(header)
            f.writelines(buf)
        cand = vs.validate_id_list_file(tmp, vs.CANDIDATE_HEADER, "candidate_entity_ids",
                                        ids & required, valid, errors)
        if matched is not None and cand is not None:
            offenders |= {s for s, c in cand.items() if matched.get(s, set()) - c}
        n += 1
    os.remove(tmp)
    for bad, msg in ((dups, "duplicate S1 rows across file"), (required - seen, "required S1 missing"),
                     (seen - required, "S1 IDs not in test set")):
        if bad:
            errors.append(f"candidate_pairs.tsv: {msg}: {vs.examples(bad)}")
    if matched is not None:
        missing = {s for s, m in matched.items() if m and s not in seen}
        if offenders or missing:
            warnings.append(f"{len(offenders | missing)} S1 have matched IDs not in candidate_pairs.tsv")
    print(f"  candidate_pairs.tsv: {len(seen)} rows in {n} chunks")
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)
    print("RESULT:", "FAIL" if errors else "PASS")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
