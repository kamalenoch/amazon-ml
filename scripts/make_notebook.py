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

v2 pipeline: normalize → key index (cap 1200/600) → learned pre-ranker pruning → noise-model + string features → LightGBM → one-to-one selection.
Settings: **Accelerator: None (CPU)**, **Internet: On**. Add the challenge data as an Input.
Run with **Save Version → Save & Run All (Commit)** — only committed versions keep their output files.
Outputs kept for the team: `output/*.tsv`, `work/model.pkl`, `work/prerank.txt`, `work/holdout_preds.parquet`,
`work/test_scores.parquet`, `work/*_cands.parquet`, `work/*_norm_*.parquet`, `work/train_report.json`, `run.log`.
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
STAGE = os.environ.get("STAGE", "all")
if STAGE == "predict":
    code("""
# Job F: attach the outputs of job T (model.pkl, prerank.txt) and job S (test_cands.parquet, test_norm_*)
imp = sorted({os.path.dirname(p) for pat in ("model.pkl", "prerank.txt", "test_cands.parquet", "test_norm_s1.parquet")
              for p in glob.glob(f"/kaggle/input/**/{pat}", recursive=True)})
print(imp)
EXTRA = " ".join(f'"{d}"' for d in imp)
""")
    code("""
!python -m src.run --data "$DATA" --work /kaggle/working/work --out /kaggle/working/output --stage predict --extra-work {EXTRA} 2>&1 | grep -v "Warning\\]" | tee /kaggle/working/run.log
""")
else:
    code(f"""
!python -m src.run --data "$DATA" --work /kaggle/working/work --out /kaggle/working/output --stage {STAGE} 2>&1 | grep -v "Warning\\\\]" | tee /kaggle/working/run.log
""")
if STAGE in ("all", "predict"):
    code("""
!python "$ROOT/utils/validate_submission.py" --matching /kaggle/working/output/matching_results.tsv \\
    --candidate /kaggle/working/output/candidate_pairs.tsv --test-dir "$DATA/test"
""")
code("""
!rm -rf /kaggle/working/ber/src/__pycache__
!du -sh /kaggle/working/*; ls -la /kaggle/working/work /kaggle/working/output 2>/dev/null
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
name = {"all": "kaggle_pipeline", "train": "kaggle_T_train", "test-cands": "kaggle_S_test_cands",
        "predict": "kaggle_F_predict"}[STAGE]
json.dump(nb, open(f"notebooks/{name}.ipynb", "w"), indent=1)
print(f"wrote notebooks/{name}.ipynb", len(cells), "cells")
