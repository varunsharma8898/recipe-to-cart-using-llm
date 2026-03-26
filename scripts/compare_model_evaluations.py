#!/usr/bin/env python3
"""
Read efficiency.json (and optionally other outputs) from output/evaluation/<model_slug>/
and print a comparison table. Run after evaluating with --model for each of gemma3:1b, gemma3:4b, gemma3:12b.

Usage:
  python scripts/compare_model_evaluations.py
  python scripts/compare_model_evaluations.py --quick  # mention quick run in output

Expects directories:
  output/evaluation/gemma3_1b/
  output/evaluation/gemma3_4b/
  output/evaluation/gemma3_12b/
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description="Compare evaluation results across LLM models.")
    ap.add_argument("--quick", action="store_true", help="Note that runs used --quick")
    args = ap.parse_args()

    models = [
        ("gemma3:1b", "gemma3_1b"),
        ("gemma3:4b", "gemma3_4b"),
        ("gemma3:12b", "gemma3_12b"),
    ]
    out_dir = ROOT / "output" / "evaluation"
    rows = []
    for label, slug in models:
        path = out_dir / slug / "efficiency.json"
        if not path.exists():
            print(f"Warning: {path} not found. Run: python scripts/run_evaluation.py --model {label} [--quick]", file=sys.stderr)
            rows.append((label, None))
            continue
        with open(path) as f:
            data = json.load(f)
        rows.append((label, data))

    # Table: model, mean_latency_seconds, mean_time_per_ingredient_seconds, total_seconds, n_recipes
    print("## Model comparison (efficiency)")
    if args.quick:
        print("(Evaluations run with --quick: 3 recipes for efficiency.)")
    print()
    print("| Model       | Mean latency/recipe (s) | Mean time/ingredient (s) | Total (s) | N recipes |")
    print("|-------------|-------------------------|---------------------------|-----------|-----------|")
    for label, data in rows:
        if data is None:
            print(f"| {label:11} | (missing)               | (missing)                 |           |           |")
            continue
        n = data.get("n_recipes", "")
        total = data.get("total_seconds", "")
        mean_rec = data.get("mean_latency_seconds", "")
        mean_ing = data.get("mean_time_per_ingredient_seconds", "")
        if isinstance(mean_rec, (int, float)) and isinstance(mean_ing, (int, float)):
            print(f"| {label:11} | {mean_rec:23.2f} | {mean_ing:25.2f} | {total:9.2f} | {n:9} |")
        else:
            print(f"| {label:11} | {str(mean_rec):23} | {str(mean_ing):25} | {str(total):9} | {str(n):9} |")

    print()
    print("Output directories:")
    for label, slug in models:
        d = out_dir / slug
        print(f"  - {label}: {d}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
