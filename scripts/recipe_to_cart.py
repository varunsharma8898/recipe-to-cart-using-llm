#!/usr/bin/env python3
"""
Build a shopping cart from a recipe name (retrieve → re-rank → quantity optimise).
Usage: python scripts/recipe_to_cart.py [recipe_name]
If recipe_name is omitted, uses first recipe in the dataset.
Requires: prepare_data output (products_food.csv), Ollama with gemma3:4b for re-ranker.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.load_recipes import load_recipes, RECIPE_NAME_COL
from src.build_cart import build_cart, cart_to_rows, get_build_cart_params


def main():
    df = load_recipes()
    if len(sys.argv) > 1:
        recipe_name = " ".join(sys.argv[1:])
    else:
        recipe_name = df[RECIPE_NAME_COL].iloc[0]
    params = get_build_cart_params()
    top_k = int(params.get("top_k_retrieve", 10))
    max_rerank = int(params.get("max_candidates_rerank", 5))
    use_reranker = bool(params.get("use_reranker", True))
    retrieval_method = str(params.get("retrieval_method", "bm25")).strip().lower()
    if retrieval_method != "llm":
        retrieval_method = "bm25"
    llm_pool = int(params.get("llm_retrieval_pool_size", 30))
    print("Recipe:", recipe_name)
    print("Config: retrieval=", retrieval_method, ", reranker=", "on" if use_reranker else "off", sep="")
    print("Building cart (retrieve + re-rank + optimise)...")
    cart, recipe_row, ingredients = build_cart(
        recipe_name,
        df_recipes=df,
        top_k_retrieve=top_k,
        max_candidates_rerank=max_rerank,
        use_reranker=use_reranker,
        retrieval_method=retrieval_method,
        llm_retrieval_pool_size=llm_pool,
    )
    print(f"Ingredients: {len(ingredients)} → Cart items: {len(cart)}")
    rows = cart_to_rows(cart)
    total = sum(r["total_price"] for r in rows)
    for i, r in enumerate(rows, 1):
        print(f"  {i}. {r['product'][:50]} | {r['quantity']} x {r['sale_price']} = {r['total_price']}")
    print(f"Cart total: {total:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
