"""Build a self-contained Kaggle notebook: writes src/ via %%writefile, runs the pipeline.
Run from the repo root: python scripts/make_notebook.py"""
import json
import glob
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# Amazon ML Challenge 2026 — Business Entity Resolution

Pipeline: normalize → multi-key IDF blocking → pair features → LightGBM → one-to-one selection.
Settings: **Accelerator: None (CPU)**, **Internet: On**. Add the challenge data as an Input.
Use **Save Version → Save & Run All (Commit)** so it runs in the background (≈1.5–2.5 h).
""")
code("""
# Offline-safe install: use wheels from an attached dataset if present, else PyPI (needs Internet On)
import glob, os, sys, subprocess
whls = glob.glob("/kaggle/input/**/*.whl", recursive=True)
try:
    import unidecode, rapidfuzz
    print("already installed")
except ImportError:
    if whls:
        dirs = sorted({os.path.dirname(w) for w in whls})
        cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-index"] + sum([["--find-links", d] for d in dirs], []) + ["unidecode", "rapidfuzz"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-q", "unidecode==1.3.8", "rapidfuzz==3.9.7"]
    print(subprocess.run(cmd, capture_output=True, text=True).stderr[-2000:])
import unidecode, rapidfuzz, lightgbm, pandas, numpy
print("python", sys.version.split()[0], "rapidfuzz", rapidfuzz.__version__, "lightgbm", lightgbm.__version__, "pandas", pandas.__version__, "numpy", numpy.__version__)
!nproc; free -g
""")
code("""
os.makedirs("/kaggle/working/ber/src", exist_ok=True)
%cd /kaggle/working/ber
""")
for f in sorted(glob.glob("src/*.py")):
    code(f"%%writefile {f}\n" + open(f).read())
if not os.path.exists("src/__init__.py"):
    code("%%writefile src/__init__.py\n\n")
code("""
hits = glob.glob("/kaggle/input/**/train_source1.tsv", recursive=True)
assert hits, "Add the challenge dataset as an Input first"
DATA = os.path.dirname(os.path.dirname(hits[0]))   # .../dataset
ROOT = os.path.dirname(DATA)                       # .../student_resource
print(DATA)
""")
code("""
!python -m src.run --data "$DATA" --work /kaggle/working/work --out /kaggle/working/output 2>&1 | grep -v "Warning\\]"
""")
code("""
!python "$ROOT/utils/validate_submission.py" --matching /kaggle/working/output/matching_results.tsv \\
    --candidate /kaggle/working/output/candidate_pairs.tsv --test-dir "$DATA/test"
!cat /kaggle/working/work/train_report.json | head -5
""")
code("""
# free disk: keep outputs + model/report, drop big caches
!rm -f /kaggle/working/work/*_norm_*.parquet /kaggle/working/work/*_cands.parquet
!ls -la /kaggle/working/output /kaggle/working/work
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open("notebooks/kaggle_pipeline.ipynb", "w"), indent=1)
print("wrote notebooks/kaggle_pipeline.ipynb", len(cells), "cells")
