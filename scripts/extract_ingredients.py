#!/usr/bin/env python3
"""
Extract ingredients from a recipe name (LLM + normalisation).
Usage: python scripts/extract_ingredients.py [recipe_name]
If recipe_name is omitted, runs on the first recipe in the dataset.
Requires: Ollama running with model gemma3:4b (config/ollama.yaml).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.load_recipes import load_recipes, RECIPE_NAME_COL
from src.recipe_to_ingredients import recipe_name_to_ingredients


def main():
    df = load_recipes()
    if len(sys.argv) > 1:
        recipe_name = " ".join(sys.argv[1:])
    else:
        recipe_name = df[RECIPE_NAME_COL].iloc[0]
    print("Recipe:", recipe_name)
    print("Extracting ingredients (LLM + normaliser)...")
    ingredients, row = recipe_name_to_ingredients(
        recipe_name,
        df_recipes=df,
        include_instructions=True,
        normalise=True,
        use_fallback_on_failure=True,
    )
    print(f"Extracted: {len(ingredients)} ingredients")
    for i, ing in enumerate(ingredients, 1):
        print(f"  {i}. {ing['ingredient']} — {ing['quantity']} {ing['unit']}")
    if row is None:
        print("(No matching recipe row.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
