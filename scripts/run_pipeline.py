#!/usr/bin/env python3
"""
Single pipeline entry point.
Recipe name(s) → cart (JSON/CSV). Multiple recipes → one merged cart (same product, combined quantity).
Optional: run evaluation.

Usage:
  python scripts/run_pipeline.py --recipe "Masala Karela Recipe" --cart-out output/carts/cart.json
  python scripts/run_pipeline.py --recipe "Recipe A" --recipe "Recipe B"   # single merged cart
  python scripts/run_pipeline.py --recipe "Recipe Name" --eval --quick
  python scripts/run_pipeline.py --list-recipes 5
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Recipe-to-cart pipeline: recipe name → cart (optional: run evaluation)."
    )
    parser.add_argument(
        "--recipe",
        type=str,
        action="append",
        default=[],
        help="Recipe name (match to TranslatedRecipeName). Can be repeated for multiple recipes.",
    )
    parser.add_argument(
        "--cart-out",
        type=str,
        default=None,
        help="Output path for cart (JSON or CSV). Default: output/carts/cart.json",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv"],
        default="json",
        help="Cart output format (default: json).",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip LLM re-ranker (use BM25 top-1 only; faster).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation after cart (mAP@k, failures, efficiency, quantity sample).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="With --eval: use fewer recipes (10 mAP, 3 efficiency, 5 quantity).",
    )
    parser.add_argument(
        "--list-recipes",
        type=int,
        metavar="N",
        default=0,
        help="Print N sample recipe names and exit (no cart).",
    )
    parser.add_argument(
        "--include-ingredients",
        action="store_true",
        help="Include full ingredients list in JSON per_recipe (default: only counts; use for debugging).",
    )
    args = parser.parse_args()

    if args.list_recipes:
        from src.load_recipes import load_recipes, RECIPE_NAME_COL
        df = load_recipes()
        n = min(args.list_recipes, len(df))
        for i in range(n):
            print(df[RECIPE_NAME_COL].iloc[i])
        return 0

    if not args.recipe:
        # Default: first recipe
        from src.load_recipes import load_recipes, RECIPE_NAME_COL
        df = load_recipes()
        args.recipe = [df[RECIPE_NAME_COL].iloc[0]]
        print("No --recipe given; using first recipe:", args.recipe[0])

    from src.build_cart import build_cart, build_cart_for_recipes, cart_to_rows, get_build_cart_params

    params = get_build_cart_params()
    use_reranker = not args.no_reranker
    top_k_retrieve = int(params.get("top_k_retrieve", 10))
    max_candidates_rerank = int(params.get("max_candidates_rerank", 5))
    retrieval_method = str(params.get("retrieval_method", "bm25")).strip().lower()
    if retrieval_method != "llm":
        retrieval_method = "bm25"
    llm_retrieval_pool_size = int(params.get("llm_retrieval_pool_size", 30))

    print(f"Config: retrieval={retrieval_method}, reranker={'on' if use_reranker else 'off'}", flush=True)

    if len(args.recipe) == 1:
        recipe_name = args.recipe[0]
        print("Building cart (ingredient extraction → retrieve → re-rank)...", flush=True)
        cart, recipe_row, ingredients = build_cart(
            recipe_name,
            top_k_retrieve=top_k_retrieve,
            max_candidates_rerank=max_candidates_rerank,
            use_reranker=use_reranker,
            retrieval_method=retrieval_method,
            llm_retrieval_pool_size=llm_retrieval_pool_size,
        )
        print("Cart built.", flush=True)
        rows = cart_to_rows(cart)
        total = sum(r["total_price"] for r in rows)
        print(f"Recipe: {recipe_name}")
        print(f"  Ingredients: {len(ingredients)} → Cart items: {len(rows)} | Total: {total:.2f}")
        merged_rows = rows
        per_recipe_summary = [{"recipe_name": recipe_name, "ingredients_count": len(ingredients)}]
        if args.include_ingredients:
            per_recipe_summary[0]["ingredients"] = ingredients
    else:
        # Multiple recipes → single merged cart (same product aggregated)
        print("Building cart for multiple recipes...", flush=True)
        cart, per_recipe = build_cart_for_recipes(
            args.recipe,
            top_k_retrieve=top_k_retrieve,
            max_candidates_rerank=max_candidates_rerank,
            use_reranker=use_reranker,
            retrieval_method=retrieval_method,
            llm_retrieval_pool_size=llm_retrieval_pool_size,
        )
        print("Cart built.", flush=True)
        rows = cart_to_rows(cart)
        total = sum(r["total_price"] for r in rows)
        merged_rows = rows
        print(f"Recipes ({len(args.recipe)}): {', '.join(args.recipe[:3])}{'...' if len(args.recipe) > 3 else ''}")
        for i, name in enumerate(args.recipe):
            _, ingredients = per_recipe[i]
            print(f"  {i+1}. {name}: {len(ingredients)} ingredients")
        print(f"  → Single merged cart: {len(rows)} items | Total: {total:.2f}")
        per_recipe_summary = [
            {"recipe_name": name, "ingredients_count": len(ing)}
            for (_, ing), name in zip(per_recipe, args.recipe)
        ]
        if args.include_ingredients:
            for i, (_, ing) in enumerate(per_recipe):
                per_recipe_summary[i]["ingredients"] = ing

    out_path = args.cart_out
    if out_path is None:
        out_path = ROOT / "output" / "carts" / "cart.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "csv" or str(out_path).lower().endswith(".csv"):
        import pandas as pd
        pd.DataFrame(merged_rows).to_csv(out_path, index=False)
    else:
        out_data = {
            "recipes": args.recipe,
            "cart_items_count": len(merged_rows),
            "cart_total": round(total, 2),
            "cart": merged_rows,
            "per_recipe": per_recipe_summary,
        }
        with open(out_path, "w") as f:
            json.dump(out_data, f, indent=2)
    print(f"Cart written to {out_path}")

    if args.eval:
        print("\nRunning evaluation...")
        from scripts.run_evaluation import run_evaluation
        run_evaluation(quick=args.quick)

    return 0


if __name__ == "__main__":
    sys.exit(main())
