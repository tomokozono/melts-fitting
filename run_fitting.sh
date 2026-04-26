#!/bin/bash
# Usage: ./run_fitting.sh <path/to/melts-liquid.tbl>

set -e

TBL="${1:-melts-liquid.tbl}"

if [ ! -f "$TBL" ]; then
    echo "Error: file not found: $TBL" >&2
    exit 1
fi

echo "=== Running fitting for: $TBL ==="
echo ""

echo "[1/3] H2O solubility fitting..."
uv run python fit_h2o.py "$TBL"
echo ""

echo "[2/3] Viscosity fitting..."
uv run python fit_viscosity.py "$TBL"
echo ""

echo "[3/3] Crystallinity fitting..."
uv run python fit_crystallinity.py "$TBL"
echo ""

echo "=== Done. Output saved to: $(dirname "$TBL") ==="
