"""
Ingredient Extractor: recipe text → structured list {ingredient, quantity, unit} via LLM.
Fallback: use Cleaned-Ingredients from recipe row when LLM fails or returns invalid JSON.
"""
import json
import re
from typing import Any, Optional

import pandas as pd

from .ollama_client import chat
from .load_recipes import (
    CLEANED_INGREDIENTS_COL,
    parse_cleaned_ingredients,
)


SYSTEM_PROMPT = """You are an expert ingredient extraction assistant for Indian recipes.
Extract ingredients with quantities from the recipe text below.
Output ONLY a valid JSON array of objects. Each object must have exactly these fields: "ingredient", "quantity", "unit".
- ingredient: the ingredient name only (e.g. "butter", "maida", "red chilli powder").
- quantity: a number (use 1 for "to taste", "as required", "pinch", etc.).
- unit: standard unit like "tablespoon", "teaspoon", "cup", "gram", "piece", "pinch", "to taste" (use "piece" for whole items like "1 onion").
Do not include any explanation, only the JSON array."""

FEW_SHOT_USER = """Example:
Recipe: "Add 2 tablespoons of butter and 1 cup maida. Salt to taste."
Output: [{"ingredient": "butter", "quantity": 2, "unit": "tablespoons"}, {"ingredient": "maida", "quantity": 1, "unit": "cup"}, {"ingredient": "salt", "quantity": 1, "unit": "to taste"}]

Now extract from this recipe (output only the JSON array, no other text):

"""


def _parse_llm_json(raw: str) -> list[dict[str, Any]]:
    """Extract JSON array from model output (may be wrapped in ```json ... ```)."""
    raw = raw.strip()
    # Remove markdown code block if present
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        raw = m.group(1).strip()
    # Find first [ and last ]
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(out, list):
        return []
    result = []
    for item in out:
        if not isinstance(item, dict):
            continue
        ing = item.get("ingredient") or item.get("name")
        if not ing:
            continue
        q = item.get("quantity")
        if q is None or (isinstance(q, str) and not q.strip()):
            q = 1
        try:
            q = float(q) if isinstance(q, str) else float(q)
        except (TypeError, ValueError):
            q = 1
        u = (item.get("unit") or "").strip() or "piece"
        result.append({"ingredient": str(ing).strip(), "quantity": q, "unit": str(u)})
    return result


def extract_ingredients_llm(recipe_text: str) -> list[dict[str, Any]]:
    """
    Call LLM to extract structured ingredients from recipe text.

    Parameters
    ----------
    recipe_text : str
        Full recipe text (ingredients list + optionally instructions).

    Returns
    -------
    list[dict]
        List of {"ingredient": str, "quantity": float, "unit": str}.
        Empty list on failure.
    """
    if not (recipe_text or recipe_text.strip()):
        return []
    user_content = FEW_SHOT_USER + recipe_text
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        response = chat(messages)
        return _parse_llm_json(response)
    except Exception:
        return []


def fallback_from_cleaned_ingredients(recipe_row: pd.Series) -> list[dict[str, Any]]:
    """
    Build structured list from Cleaned-Ingredients when LLM is unavailable or failed.
    No quantity/unit parsing; uses quantity=1, unit="piece".
    """
    raw = recipe_row.get(CLEANED_INGREDIENTS_COL)
    names = parse_cleaned_ingredients(raw) if pd.notna(raw) else []
    return [{"ingredient": n, "quantity": 1, "unit": "piece"} for n in names]


def extract_ingredients(
    recipe_text: str,
    recipe_row: Optional[pd.Series] = None,
    use_fallback_on_failure: bool = True,
) -> list[dict[str, Any]]:
    """
    Extract ingredients with quantities: try LLM first, then fallback to Cleaned-Ingredients.

    Parameters
    ----------
    recipe_text : str
        Recipe text for the LLM (from get_recipe_text_for_llm).
    recipe_row : pd.Series or None
        If provided and LLM fails, use this row's Cleaned-Ingredients for fallback.
    use_fallback_on_failure : bool
        If True and LLM returns empty or raises, use fallback when recipe_row is given.

    Returns
    -------
    list[dict]
        [{"ingredient", "quantity", "unit"}, ...].
    """
    out = extract_ingredients_llm(recipe_text)
    if out:
        return out
    if use_fallback_on_failure and recipe_row is not None:
        return fallback_from_cleaned_ingredients(recipe_row)
    return []
