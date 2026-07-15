#!/usr/bin/env bash
# Runs the comparative benchmark (reflective vs. non-reflective pipeline).
set -euo pipefail

CONFIG="${1:-configs/benchmark.yaml}"
python -m quantlab.eval.run_benchmark --config "${CONFIG}"
