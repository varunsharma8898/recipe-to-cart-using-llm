#!/usr/bin/env python3
"""
Run evaluation: mAP@k, efficiency benchmark, failure analysis, quantity verification.
Writes results to output/evaluation/ or output/evaluation/<model_slug>/ when --model is set.

Usage:
  python scripts/run_evaluation.py [--quick]
  python scripts/run_evaluation.py --model gemma3:1b [--quick]
  python scripts/run_evaluation.py --model gemma3:4b [--quick]
  python scripts/run_evaluation.py --model gemma3:12b [--quick]

  --quick: use 10 recipes for mAP and 3 for efficiency/quantity (faster).
  --model: Ollama model name (e.g. gemma3:1b, gemma3:4b, gemma3:12b). Sets OLLAMA_MODEL
           and writes to output/evaluation/<model_slug>/ for comparison.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import (
    load_evaluation_set,
    run_map_evaluation,
    collect_failures,
    run_efficiency_benchmark,
    run_quantity_verification_sample,
    CandidateRetriever,
    load_food_products_cached,
)


def _model_slug(model: str | None) -> str:
    """Convert e.g. gemma3:4b -> gemma3_4b for directory names."""
    if not model:
        return "default"
    return model.replace(":", "_").replace("/", "_")


def run_evaluation(quick: bool = False, model: str | None = None) -> None:
    """Run full evaluation; write results to output/evaluation/ or output/evaluation/<model_slug>/ when --model is set."""
    if model:
        slug = _model_slug(model)
        out_dir = ROOT / "output" / "evaluation" / slug
    else:
        out_dir = ROOT / "output" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_recipes = 10 if quick else 50
    print("Evaluation" + (f" (model={model})" if model else ""))
    print("=" * 50)

    # 1. mAP@k
    print("\n1. mAP@k (retrieval evaluation)...")
    results = run_map_evaluation(k_values=[1, 5, 10], max_recipes=max_recipes)
    for k in [1, 5, 10]:
        print(f"   mAP@{k} = {results[f'mAP@{k}']:.4f}")
        print(f"   (n_queries={results[f'stats@{k}']['n_queries']}, no_candidates={results[f'stats@{k}']['n_no_candidates']})")
    def _to_json_serializable(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_json_serializable(x) for x in obj]
        return obj

    with open(out_dir / "map_results.json", "w") as f:
        json.dump(_to_json_serializable({k: v for k, v in results.items() if not k.startswith("stats@")}), f, indent=2)
    with open(out_dir / "map_stats.json", "w") as f:
        json.dump(_to_json_serializable({f"stats@{k}": results.get(f"stats@{k}") for k in [1, 5, 10]}), f, indent=2)
    print(f"   Saved {out_dir / 'map_results.json'}")

    # 2. Failure analysis (unmatched ingredients)
    print("\n2. Failure analysis (ingredients with no BM25 candidates)...")
    eval_df = load_evaluation_set()
    df_food = load_food_products_cached()
    retriever = CandidateRetriever(df_products=df_food, use_food_filter=False)
    failures = collect_failures(eval_df, retriever, k=10, max_recipes=max_recipes)
    print(f"   Unmatched ingredients: {len(failures)}")
    failure_recipes = set(f["recipe_name"] for f in failures[:20])
    for f in failures[:15]:
        print(f"     - {f['ingredient'][:40]} ({f['recipe_name'][:35]})")
    pd.DataFrame(failures).to_csv(out_dir / "failures_unmatched_ingredients.csv", index=False)
    print(f"   Saved {out_dir / 'failures_unmatched_ingredients.csv'}")

    # 3. Efficiency
    print("\n3. Efficiency benchmark (build_cart latency)...")
    n_eff = 3 if quick else 5
    eff = run_efficiency_benchmark(n_recipes=n_eff, use_reranker=True)
    if model:
        eff["model"] = model
    print(f"   Mean latency per recipe: {eff['mean_latency_seconds']:.2f} s")
    print(f"   Mean time per ingredient: {eff['mean_time_per_ingredient_seconds']:.2f} s")
    with open(out_dir / "efficiency.json", "w") as f:
        json.dump(eff, f, indent=2)
    print(f"   Saved {out_dir / 'efficiency.json'}")

    # 4. Quantity verification sample (for manual check)
    print("\n4. Quantity verification sample (salt/sugar)...")
    n_qty = 5 if quick else 15
    qty_rows = run_quantity_verification_sample(n_recipes=n_qty, ingredients_to_check=["salt", "sugar"])
    pd.DataFrame(qty_rows).to_csv(out_dir / "quantity_verification_sample.csv", index=False)
    print(f"   Rows: {len(qty_rows)} (saved for manual verification)")
    print(f"   Saved {out_dir / 'quantity_verification_sample.csv'}")

    print(f"\nDone. Results in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Run evaluation (mAP, failures, efficiency, quantity verification).")
    ap.add_argument("--quick", action="store_true", help="Fewer recipes for faster run")
    ap.add_argument("--model", type=str, default=None,
                    help="Ollama model name (e.g. gemma3:1b, gemma3:4b, gemma3:12b). Results written to output/evaluation/<model_slug>/")
    args = ap.parse_args()
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model
    run_evaluation(quick=args.quick, model=args.model)


if __name__ == "__main__":
    main()
