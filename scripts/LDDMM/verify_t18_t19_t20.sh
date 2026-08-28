#!/bin/bash
# Verification script for T18, T19, T20
# Run from project root: bash scripts/verify_t18_t19_t20.sh

set -e

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "T18: Callback compatibility widening"
echo "=========================================="
python -m pytest tests/learning/callbacks/test_callbacks_compat.py -xvs

echo ""
echo "=========================================="
echo "T19: Diagnostics functions"
echo "=========================================="
python -m pytest tests/resnet_lddmm/test_diagnostics.py -xvs

echo ""
echo "=========================================="
echo "T20: Diagnostics and export callbacks"
echo "=========================================="
python -m pytest tests/resnet_lddmm/test_callbacks.py -xvs

echo ""
echo "=========================================="
echo "Full ResNetLDDMM + learning callbacks suite"
echo "=========================================="
python -m pytest tests/resnet_lddmm/ tests/learning/callbacks/ -q

echo ""
echo "✓ All T18/T19/T20 tests pass!"
