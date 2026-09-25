#!/usr/bin/env bash
# Build the final submission zip: output/ + code/business_entity_resolution/ + Documentation_template.md
# Usage: scripts/package_submission.sh <team_name> <dir with matching_results.tsv & candidate_pairs.tsv>
set -euo pipefail
TEAM=$1; OUT=$2; T=$(mktemp -d)
mkdir -p "$T/output" "$T/code/business_entity_resolution"
cp "$OUT/matching_results.tsv" "$OUT/candidate_pairs.tsv" "$T/output/"
cp -r src README.md requirements.txt "$T/code/business_entity_resolution/"
cp notebooks/kaggle_pipeline.ipynb "$T/code/business_entity_resolution/"
cp docs/Documentation_template.md "$T/Documentation_template.md"
find "$T" -name __pycache__ -prune -exec rm -rf {} +
(cd "$T" && zip -qr "$OLDPWD/${TEAM}_submission.zip" .)
echo "wrote ${TEAM}_submission.zip"
