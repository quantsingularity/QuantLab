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
    echo "Note: OPENAI_API_KEY is not set. The PoC agents are deterministic" >&2
    echo "stubs and do not require it yet, but core/llm.py will need it once" >&2
    echo "v0.5 wires in real LLM calls." >&2
fi

echo "Running QuantLab PoC on: \"${OBJECTIVE}\""
python -m quantlab.run --objective "${OBJECTIVE}" --out "${OUT_DIR}"
echo "Artefacts written to ${OUT_DIR}"
