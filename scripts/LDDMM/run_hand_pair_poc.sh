#!/bin/bash
# Run hand pair registration proof-of-concept
# Registers template.vtp to itself to verify correctness

set -e

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "Running hand pair PoC registration"
echo "=========================================="
python -m src.resnet_lddmm.cli configs/hand_pair_poc.yaml

echo ""
echo "✓ Training complete!"
echo "Outputs saved to: outputs/hand_pair_poc/"
