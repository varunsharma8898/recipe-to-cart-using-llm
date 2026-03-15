"""
Quantity Normaliser: canonical ingredient names, unit standardisation, disambiguation via LLM.
Input: list of {ingredient, quantity, unit} from ingredient_extractor.
Output: same structure with normalised ingredient names and units.
"""
import json
import re
from typing import Any

from .ollama_client import chat


SYSTEM_PROMPT = """You are a recipe ingredient normalisation assistant for Indian grocery shopping.
Given a list of ingredients with quantities and units, output a normalised list in the same JSON format.
Tasks:
1. Use canonical ingredient names (e.g. "maida" -> "all-purpose flour", "dhania" -> "coriander", "besan" -> "gram flour").
2. Map regional/Indian names to a standard name where helpful (e.g. Catla -> catla, Rohu -> rohu, Ilish -> hilsa, Paplet -> pomfret).
3. Resolve ambiguous terms (e.g. "oil" -> "vegetable oil", "green chilli" -> "green chilli").
4. Standardise units: use "tablespoon", "teaspoon", "cup", "gram", "piece", "pinch", "to taste", "ml" as appropriate.
5. Keep quantity as a number. Keep the same number of items; only change names and units.
6. When clear from the name, add a "form" field to help product matching: use "base" for whole/raw (vegetable, nuts, leaves, whole spice), "powder" for powdered, "juice" for juice, "butter" for butter/nut butter, "paste" for paste, "leaves" for leaves, "oil" for oil. Omit "form" if unsure (will be treated as base).
Output ONLY a valid JSON array of objects with fields: "ingredient", "quantity", "unit", and optionally "form". No explanation."""


def _parse_normaliser_json(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        raw = m.group(1).strip()
    start, end = raw.find("["), raw.rfind("]")
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
        ing = (item.get("ingredient") or item.get("name") or "").strip()
        if not ing:
            continue
        q = item.get("quantity", 1)
        try:
            q = float(q)
        except (TypeError, ValueError):
            q = 1
        u = (item.get("unit") or "piece").strip() or "piece"
        form = (item.get("form") or "base").strip().lower() or "base"
        if form not in ("base", "powder", "juice", "butter", "paste", "leaves", "oil"):
            form = "base"
        result.append({"ingredient": ing, "quantity": q, "unit": u, "form": form})
    return result


def normalise_ingredients(ingredients: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Call LLM to normalise ingredient names and units.

    Parameters
    ----------
    ingredients : list[dict]
        From ingredient_extractor: [{"ingredient", "quantity", "unit"}, ...].

    Returns
    -------
    list[dict]
        Same structure with normalised ingredient names and units.
        Returns input list unchanged if LLM fails or returns invalid JSON.
    """
    if not ingredients:
        return []
    # Build compact text for the model
    lines = [f"- {x['ingredient']}: {x['quantity']} {x['unit']}" for x in ingredients]
    user_content = "Normalise these ingredients (output only the JSON array):\n\n" + "\n".join(lines)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        response = chat(messages)
        out = _parse_normaliser_json(response)
        if out:
            return out
    except Exception:
        pass
    return ingredients
