#!/usr/bin/env bash
# Quickstart wrapper for the two-day PoC.
#
# Runs a basic sanity check (dependencies installed) and then invokes the
# nine-agent pipeline on the default momentum objective.
set -euo pipefail

OBJECTIVE="${1:-Develop a momentum strategy for the NASDAQ 100}"
OUT_DIR="${2:-./runs/momentum_nasdaq}"

if ! python -c "import quantlab" >/dev/null 2>&1; then
    echo "quantlab is not installed. Run: pip install -e \".[dev]\"" >&2
    exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Note: OPENAI_API_KEY is not set. Every agent has a deterministic" >&2
    echo "fallback, so the run below will use those; set OPENAI_API_KEY and" >&2
    echo "name a model under a config's models section to draft text with" >&2
    echo "an LLM instead." >&2
fi

echo "Running QuantLab PoC on: \"${OBJECTIVE}\""
python -m quantlab.run --objective "${OBJECTIVE}" --out "${OUT_DIR}"
echo "Artefacts written to ${OUT_DIR}"
