"""
Pipeline: recipe name → recipe text → LLM ingredient extraction → optional normalisation.
Single entry point for "recipe name to structured ingredients".
"""
from typing import Any, Optional

import pandas as pd

from .load_recipes import (
    load_recipes,
    get_recipe_by_name,
    get_recipe_text_for_llm,
    RECIPE_NAME_COL,
)
from .ingredient_extractor import extract_ingredients
from .quantity_normaliser import normalise_ingredients


def recipe_name_to_ingredients(
    recipe_name: str,
    df_recipes: Optional[pd.DataFrame] = None,
    include_instructions: bool = True,
    normalise: bool = True,
    use_fallback_on_failure: bool = True,
) -> tuple[list[dict[str, Any]], Optional[pd.Series]]:
    """
    From recipe name to structured ingredient list (with optional normalisation).

    Parameters
    ----------
    recipe_name : str
        Exact or case-insensitive match to TranslatedRecipeName.
    df_recipes : pd.DataFrame or None
        Recipe dataframe; if None, load_recipes() is called.
    include_instructions : bool
        Passed to get_recipe_text_for_llm.
    normalise : bool
        If True, run quantity_normaliser on extracted ingredients.
    use_fallback_on_failure : bool
        If True, use Cleaned-Ingredients when LLM extraction returns empty.

    Returns
    -------
    ingredients : list[dict]
        [{"ingredient", "quantity", "unit"}, ...].
    recipe_row : pd.Series or None
        The matched recipe row, or None if not found.
    """
    if df_recipes is None:
        df_recipes = load_recipes()
    row = get_recipe_by_name(df_recipes, recipe_name)
    if row is None:
        return [], None
    recipe_text = get_recipe_text_for_llm(row, include_instructions=include_instructions)
    ingredients = extract_ingredients(
        recipe_text,
        recipe_row=row,
        use_fallback_on_failure=use_fallback_on_failure,
    )
    if normalise and ingredients:
        ingredients = normalise_ingredients(ingredients)
    return ingredients, row


if __name__ == "__main__":
    # Quick test: first recipe name from data
    df = load_recipes()
    name = df[RECIPE_NAME_COL].iloc[0]
    print("Recipe:", name)
    ingredients, _ = recipe_name_to_ingredients(name, df_recipes=df, normalise=True)
    print("Extracted ingredients:", ingredients[:5])
