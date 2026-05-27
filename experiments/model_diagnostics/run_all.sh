#!/bin/bash
# Run diagnostics for a curated subset of generators.
# Usage: bash run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="../../venv/Scripts/python"

GENERATORS=(
    "Matalas"
    "ThomasFiering"
    "GaussianCopula"
    "Kirsch"
    "ARFIMA"
    "SMARTA"
    "SPARTA"
)

for gen in "${GENERATORS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Running diagnostic: $gen"
    echo "============================================================"
    $PYTHON run_diagnostic.py --generator "$gen" --n_realizations 3 --n_years 30
done

echo ""
echo "All diagnostics complete. Results in outputs/"
