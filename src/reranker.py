"""
Re-ranker: LLM ranks BM25 candidates for an ingredient given recipe context.
"""
import json
import re
from typing import Any, Optional

from .ollama_client import chat


def _format_candidates(candidates: list[dict[str, Any]], max_items: int = 10) -> str:
    """Format candidate list for the prompt."""
    lines = []
    for i, c in enumerate(candidates[:max_items], 1):
        name = c.get("product", "")
        brand = c.get("brand", "")
        price = c.get("sale_price", "")
        rating = c.get("rating", "")
        lines.append(f"{i}. {name} (brand: {brand}, price: {price}, rating: {rating})")
    return "\n".join(lines)


def _parse_top_k_indices(raw: str, n_candidates: int) -> Optional[list[int]]:
    """
    Parse LLM output for top-k selection: expect JSON with "product_indices" or "ranked_indices" (1-based).
    Returns list of 1-based indices, or None on parse failure.
    """
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        raw = m.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a JSON array of numbers
        arr_m = re.search(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", raw)
        if arr_m:
            try:
                indices = json.loads(arr_m.group(0))
                if isinstance(indices, list) and all(isinstance(i, int) for i in indices):
                    return [int(i) for i in indices if 1 <= i <= n_candidates]
            except (json.JSONDecodeError, TypeError):
                pass
        return None
    if not isinstance(data, dict):
        return None
    indices = data.get("product_indices") or data.get("ranked_indices") or data.get("indices")
    if indices is None or not isinstance(indices, list):
        return None
    out = []
    seen = set()
    for i in indices:
        try:
            idx = int(i)
            if 1 <= idx <= n_candidates and idx not in seen:
                out.append(idx)
                seen.add(idx)
        except (TypeError, ValueError):
            continue
    return out if out else None


def llm_select_top_k(
    ingredient: str,
    candidates: list[dict[str, Any]],
    k: int,
    recipe_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Use LLM to select and order the top-k products from a candidate pool for this ingredient.
    Used when retrieval_method is "llm": BM25 gives a pool, this returns the best k.

    Parameters
    ----------
    ingredient : str
        Normalised ingredient name.
    candidates : list[dict]
        Candidate products (e.g. from BM25 with larger k).
    k : int
        Number of products to return (ordered best first).
    recipe_name : str or None
        Optional recipe name for context.

    Returns
    -------
    list[dict]
        Up to k candidates in LLM-ranked order. Falls back to candidates[:k] on parse failure.
    """
    if not candidates:
        return []
    if len(candidates) <= k:
        return list(candidates)
    text = _format_candidates(candidates, max_items=len(candidates))
    system = (
        "You are a product recommendation expert for Indian grocery shopping. "
        "Given an ingredient and a numbered list of candidate products, return the top-k best matches "
        "in order of relevance (best first). Use the same rules as picking the single best: "
        "prefer base/whole form, avoid juice/butter/pickle when ingredient is vegetable/fruit/nut, "
        "avoid ready-to-cook/snacks for staples. "
        "Reply with ONLY a JSON object with one key: \"product_indices\" — an array of 1-based "
        "product numbers in order of preference (best first), e.g. {\"product_indices\": [3, 1, 5, 2]}."
    )
    user = (
        f"Recipe: {recipe_name or 'N/A'}\n"
        f"Ingredient needed: {ingredient}\n\n"
        f"Candidate products:\n{text}\n\n"
        f"Return the top {k} product numbers (1-based) in order of best match. JSON only: {{\"product_indices\": [ ... ]}}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        response = chat(messages)
        indices = _parse_top_k_indices(response, len(candidates))
        if indices:
            return [candidates[i - 1] for i in indices[:k]]
    except Exception:
        pass
    return candidates[:k]


def _parse_reranker_output(raw: str, candidates: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """
    Parse LLM output: expect JSON with "rank" or "product_index" or "best" (1-based).
    Return the winning candidate dict or None.
    """
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        raw = m.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find "1" or "product_index": 1 or "rank": 1
        for pattern in [r'"product_index"\s*:\s*(\d+)', r'"rank"\s*:\s*(\d+)', r'"best"\s*:\s*(\d+)', r'(\d+)\s*\)']:
            match = re.search(pattern, raw)
            if match:
                idx = int(match.group(1))
                if 1 <= idx <= len(candidates):
                    return candidates[idx - 1]
        return None
    if not isinstance(data, dict):
        return None
    idx = data.get("product_index") or data.get("rank") or data.get("best")
    if idx is not None and 1 <= idx <= len(candidates):
        return candidates[idx - 1]
    return None


def rerank(
    ingredient: str,
    quantity: float,
    unit: str,
    recipe_name: str,
    candidates: list[dict[str, Any]],
    max_candidates: int = 5,
    form_hint: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Use LLM to pick the best product from BM25 candidates for this ingredient.

    Parameters
    ----------
    ingredient : str
        Normalised ingredient name.
    quantity : float
        Quantity needed.
    unit : str
        Unit (e.g. tablespoon, cup).
    recipe_name : str
        Recipe name for context.
    candidates : list[dict]
        From CandidateRetriever.retrieve(ingredient, k=...).
    max_candidates : int
        How many candidates to send to the LLM (to limit prompt size).
    form_hint : str or None
        Optional "base" | "powder" | "juice" | "butter" | "paste" | "leaves". When "base",
        prefer whole/raw product; avoid juice/butter/powder unless recipe needs it.

    Returns
    -------
    dict or None
        Best candidate (same structure as list items) or None if parsing fails.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    text = _format_candidates(candidates, max_items=max_candidates)
    system = (
        "You are a product recommendation expert for Indian grocery shopping. "
        "Given an ingredient and a list of candidate products, choose the single best product. "
        "Important rules: "
        "(1) Prefer the BASE/WHOLE form of the ingredient (e.g. vegetable, raw nuts, whole spice, leaves) "
        "unless the recipe or ingredient name clearly asks for a processed form (e.g. 'moringa powder', 'cashew butter'). "
        "(2) Do NOT pick juice when the ingredient is a vegetable or fruit name (e.g. karela → bitter gourd vegetable, not karela juice). "
        "(3) Do NOT pick butter or spread when the ingredient is nuts or seeds (e.g. cashews → cashew nuts, not cashew butter). "
        "(4) Do NOT pick pickle/achar/pickled when the ingredient is a vegetable or chilli (e.g. karela → fresh bitter gourd, not karela achar; green chillies → fresh, not pickled green chillies). "
        "(5) Do NOT pick cookies, biscuits, or snacks when the ingredient is a nut (e.g. cashews → raw cashew nuts, NOT cashew cookies or biscuit). "
        "(6) Do NOT pick ready-to-eat dishes (e.g. upma, instant mix) when the ingredient is a cooking staple: vegetable oil → cooking oil (prefer sunflower/refined oil), NOT vegetable upma or mix; rice → raw rice grain, NOT rice bran oil or oil pouch. "
        "(7) For 'rice' or 'raw rice' choose raw rice / rice grain; do NOT choose rice bran oil, cooking oil, or oil pouches. "
        "(8) For 'vegetable oil' or 'cooking oil' choose edible cooking oil (prefer sunflower oil if available); do NOT choose upma, instant mix, or breakfast mixes. "
        "(9) For tomato/tomatoes/potato/potatoes/aloo/onion choose the RAW vegetable (e.g. potato, onion); do NOT choose ready-to-cook, paratha, tikki, frozen snacks (e.g. Aloo Paratha, Aloo Tikki, McCain). "
        "(10) For flour / maida / all-purpose flour choose actual flour (atta, refined flour, maida); do NOT choose breakfast cereal, flakes, All Bran, or muesli. "
        "(11) For MAIDA / all-purpose flour / refined flour: choose ONLY products that are maida, all-purpose flour, or refined wheat flour. Do NOT choose besan (gram flour), rice flour, urad dal flour, or any other flour — they are different ingredients. "
        "(12) Prefer this hierarchy: base > cuts > powder > juice > paste > pickle > oil. "
        "Choose powder/juice/butter/pickle only if the recipe clearly needs that form. "
        "If an ingredient is unavailable, return the best alternative product. "
        "Also consider: recipe context, package size vs quantity needed, price, and typical household use. "
        "Reply with ONLY a JSON object with one key: \"product_index\" (1-based, the number of the best product in the list)."
    )
    prefer_base = (form_hint or "").strip().lower() == "base" or not form_hint
    form_instruction = (
        " Prefer base/whole form (e.g. vegetable, raw nuts); avoid juice, butter, powder unless the recipe clearly needs it."
        if prefer_base else f" Recipe may need form: {form_hint}."
    )
    user = (
        f"Recipe: {recipe_name}\n"
        f"Ingredient needed: {ingredient} — {quantity} {unit}\n"
        f"{form_instruction}\n\n"
        f"Candidate products:\n{text}\n\n"
        "Which product is the best match? Reply with JSON only: {\"product_index\": N}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        response = chat(messages)
        return _parse_reranker_output(response, candidates[:max_candidates])
    except Exception:
        return candidates[0]
