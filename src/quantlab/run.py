"""CLI entry point: `python -m quantlab.run --objective "..."`."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from quantlab.core.graph import build_graph
from quantlab.core.seeding import DEFAULT_SEED, apply_seed
from quantlab.core.state import ResearchState


def _to_json(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, list):
        return [_to_json(x) for x in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def main() -> None:
    parser = argparse.ArgumentParser("quantlab")
    parser.add_argument(
        "--objective",
        default=None,
        help="Natural-language research objective. Overrides --config's objective if both are given.",
    )
    parser.add_argument("--out", default="./runs", help="Output directory.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config (e.g. configs/momentum_nasdaq.yaml) supplying "
        "universe/start/end/oos_start/transaction_cost_bps.",
    )
    args = parser.parse_args()

    run_config: dict[str, Any] = {}
    if args.config:
        run_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}

    objective = args.objective or run_config.get("objective")
    if not objective:
        parser.error("--objective is required unless --config supplies one.")

    run_config.setdefault("seed", DEFAULT_SEED)
    apply_seed(int(run_config["seed"]))

    run_id = uuid.uuid4().hex[:8]
    initial: ResearchState = {
        "run_id": run_id,
        "objective": objective,
        "run_config": run_config,
        "output_dir": args.out,
        "reflections": [],
        "tokens_used": 0,
        "usd_spent": 0.0,
    }

    graph = build_graph()
    final = graph.invoke(initial)

    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {k: _to_json(v) for k, v in final.items() if not k.startswith("_")}
    (out_dir / "state.json").write_text(
        json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
    )
    print(f"Run complete. Artefacts in: {out_dir}")


if __name__ == "__main__":
    main()
