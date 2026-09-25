"""Build <team>_submission.zip with the official layout:
output/{matching_results,candidate_pairs}.tsv, code/business_entity_resolution/{src,README.md,requirements.txt},
Documentation_template.md.

Usage: python src/package.py --team <team_name>
"""
import argparse
import glob
import os
import zipfile

from common import OUT, PKG, ROOT

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    a = ap.parse_args()
    dst = os.path.join(ROOT, f"{a.team}_submission.zip")
    files = {
        "output/matching_results.tsv": os.path.join(OUT, "matching_results.tsv"),
        "output/candidate_pairs.tsv": os.path.join(OUT, "candidate_pairs.tsv"),
        "code/business_entity_resolution/README.md": os.path.join(PKG, "README.md"),
        "code/business_entity_resolution/requirements.txt": os.path.join(PKG, "requirements.txt"),
        "Documentation_template.md": os.path.join(ROOT, "Documentation_template.md"),
    }
    for p in sorted(glob.glob(os.path.join(PKG, "src", "*.py"))):
        files[f"code/business_entity_resolution/src/{os.path.basename(p)}"] = p
    missing = [k for k, v in files.items() if not os.path.exists(v)]
    assert not missing, f"missing: {missing}"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for arc, src in files.items():
            z.write(src, arc)
    print(f"wrote {dst} ({os.path.getsize(dst) / 2**20:.1f} MB, {len(files)} files)")
