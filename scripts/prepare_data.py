#!/usr/bin/env python3
"""
Load recipes and products, filter to food products, and create the evaluation set.
Run from project root: python scripts/prepare_data.py
"""
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.load_recipes import load_recipes, parse_cleaned_ingredients, RECIPE_NAME_COL, CLEANED_INGREDIENTS_COL
from src.load_products import load_products, load_food_categories, filter_food_products, _get_products_source


def main():
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading recipes...")
    df_recipes = load_recipes()
    print(f"  Recipes: {len(df_recipes)} rows")

    print("Loading products...")
    df_products = load_products()
    print(f"  Products: {len(df_products)} rows")

    food_categories = load_food_categories()
    print(f"  Food categories: {len(food_categories)}")
    df_food = filter_food_products(df_products)
    print(f"  Food products: {len(df_food)}")

    source = _get_products_source()
    food_path = out_dir / ("products_food_zepto.csv" if source == "zepto" else "products_food.csv")
    df_food.to_csv(food_path, index=False)
    print(f"Saved {food_path}")

    n_eval = 50
    cuisines = df_recipes["Cuisine"].dropna().unique()
    try:
        if len(cuisines) >= 3:
            per_cuisine = max(1, n_eval // min(len(cuisines), 10))
            eval_df = df_recipes.groupby("Cuisine", group_keys=False).apply(
                lambda g: g.sample(n=min(len(g), per_cuisine), random_state=42)
            )
            if len(eval_df) > n_eval:
                eval_df = eval_df.sample(n=n_eval, random_state=42)
            elif len(eval_df) < n_eval:
                extra = df_recipes.drop(eval_df.index).sample(n=n_eval - len(eval_df), random_state=43)
                eval_df = pd.concat([eval_df, extra])
        else:
            eval_df = df_recipes.sample(n=min(n_eval, len(df_recipes)), random_state=42)
    except Exception:
        eval_df = df_recipes.sample(n=min(n_eval, len(df_recipes)), random_state=42)
    eval_df = eval_df.head(n_eval)

    eval_export = eval_df[[RECIPE_NAME_COL, CLEANED_INGREDIENTS_COL]].copy()
    eval_export.columns = ["recipe_name", "cleaned_ingredients"]
    eval_export.insert(0, "eval_id", range(1, len(eval_export) + 1))
    eval_path = out_dir / "evaluation_set_50_recipes.csv"
    eval_export.to_csv(eval_path, index=False)
    print(f"Saved evaluation set ({len(eval_export)} recipes) to {eval_path}")

    all_ingredients = []
    for s in df_recipes[CLEANED_INGREDIENTS_COL].dropna():
        all_ingredients.extend(parse_cleaned_ingredients(s))
    unique_ingredients = len(set(all_ingredients))
    print(f"\nSummary: {len(df_recipes)} recipes, {unique_ingredients} unique ingredients; {len(df_food)} food products; {len(eval_export)} eval recipes.")


if __name__ == "__main__":
    main()
