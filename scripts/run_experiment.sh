#!/bin/bash
# Run EDRS experiments across all UCI datasets
set -e

echo "=== EDRS: Full Experiment Suite ==="
echo ""

for dataset in bank cancer thyroid sonar seismic wilt; do
    echo "--- Running: $dataset ---"
    python -m src.ssl_pipeline --config configs/config.yaml --dataset $dataset
    echo ""
done

echo "=== All experiments completed ==="
