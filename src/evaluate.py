"""
Evaluation — mAP@k, quantity verification, efficiency, failure analysis.
"""
import re
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .load_recipes import load_recipes, parse_cleaned_ingredients, get_recipe_by_name, RECIPE_NAME_COL
from .candidate_retriever import CandidateRetriever, load_food_products_cached
from .build_cart import build_cart


def _normalise_ingredient_for_relevance(ing: str) -> str:
    """Lowercase, remove parenthetical, collapse spaces."""
    if not ing:
        return ""
    s = re.sub(r"\s*\([^)]*\)\s*", " ", ing).strip().lower()
    return " ".join(s.split())


def is_product_relevant_for_ingredient(ingredient: str, product_dict: dict[str, Any]) -> bool:
    """
    Heuristic: product is relevant if ingredient (normalised) appears in product name or description.
    """
    ing_norm = _normalise_ingredient_for_relevance(ingredient)
    if not ing_norm:
        return False
    text = (
        str(product_dict.get("product", "")) + " " + str(product_dict.get("description", ""))
    ).lower()
    # Require at least the first significant token (e.g. "chilli" from "red chilli powder")
    tokens = [t for t in re.findall(r"[a-z0-9]{2,}", ing_norm)]
    if not tokens:
        return False
    return ing_norm in text or any(t in text for t in tokens)


def compute_ap_at_k(retrieved: list[dict[str, Any]], ingredient: str, k: int) -> float:
    """
    Average Precision at k for one ingredient.
    Relevance = is_product_relevant_for_ingredient(ingredient, product).
    """
    retrieved = retrieved[:k]
    if not retrieved:
        return 0.0
    relevant_positions = [
        i for i, p in enumerate(retrieved, 1) if is_product_relevant_for_ingredient(ingredient, p)
    ]
    if not relevant_positions:
        return 0.0
    # AP = (1/|relevant|) * sum_{r in relevant} precision(r)
    precisions = [len([r for r in relevant_positions if r <= pos]) / pos for pos in relevant_positions]
    return sum(precisions) / len(relevant_positions)


def load_evaluation_set(path: Optional[Path] = None) -> pd.DataFrame:
    """Load output/evaluation_set_50_recipes.csv."""
    root = Path(__file__).resolve().parent.parent
    if path is None:
        path = root / "output" / "evaluation_set_50_recipes.csv"
    if not Path(path).exists():
        raise FileNotFoundError(f"Evaluation set not found: {path}")
    return pd.read_csv(path, encoding="utf-8", on_bad_lines="warn")


def compute_map_at_k(
    eval_df: pd.DataFrame,
    retriever: CandidateRetriever,
    k: int,
    max_recipes: Optional[int] = None,
) -> tuple[float, dict[str, Any]]:
    """
    Compute mAP@k over evaluation set.
    Returns (mAP, stats_dict with per_recipe_ap, coverage, etc.).
    """
    eval_df = eval_df.head(max_recipes) if max_recipes else eval_df
    aps = []
    no_candidates_count = 0
    total_ingredients = 0
    per_recipe_ap = []
    for _, row in eval_df.iterrows():
        recipe_name = row.get("recipe_name", "")
        cleaned = row.get("cleaned_ingredients", "")
        ingredients = parse_cleaned_ingredients(cleaned)
        if not ingredients:
            continue
        recipe_aps = []
        for ing in ingredients:
            retrieved = retriever.retrieve(ing, k=k)
            total_ingredients += 1
            if not retrieved:
                no_candidates_count += 1
                recipe_aps.append(0.0)
                continue
            ap = compute_ap_at_k(retrieved, ing, k)
            recipe_aps.append(ap)
            aps.append(ap)
        if recipe_aps:
            per_recipe_ap.append(sum(recipe_aps) / len(recipe_aps))
    map_score = sum(aps) / len(aps) if aps else 0.0
    stats = {
        "mAP": map_score,
        "n_queries": len(aps),
        "n_recipes": len(per_recipe_ap),
        "n_no_candidates": no_candidates_count,
        "total_ingredients": total_ingredients,
        "mean_ap_per_recipe": sum(per_recipe_ap) / len(per_recipe_ap) if per_recipe_ap else 0.0,
    }
    return map_score, stats


def run_map_evaluation(
    k_values: list[int] = (1, 5, 10),
    max_recipes: Optional[int] = 50,
    eval_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run mAP@k for each k; return results dict."""
    eval_df = load_evaluation_set(eval_path)
    df_food = load_food_products_cached()
    retriever = CandidateRetriever(df_products=df_food, use_food_filter=False)
    results = {"k_values": k_values, "max_recipes": max_recipes or len(eval_df)}
    for k in k_values:
        map_k, stats = compute_map_at_k(eval_df, retriever, k=k, max_recipes=max_recipes)
        results[f"mAP@{k}"] = map_k
        results[f"stats@{k}"] = stats
    return results


def collect_failures(
    eval_df: pd.DataFrame,
    retriever: CandidateRetriever,
    k: int = 10,
    max_recipes: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Collect ingredients that get zero BM25 candidates (unmatched)."""
    eval_df = eval_df.head(max_recipes) if max_recipes else eval_df
    failures = []
    for _, row in eval_df.iterrows():
        recipe_name = row.get("recipe_name", "")
        ingredients = parse_cleaned_ingredients(row.get("cleaned_ingredients", ""))
        for ing in ingredients:
            retrieved = retriever.retrieve(ing, k=k)
            if not retrieved:
                failures.append({"recipe_name": recipe_name, "ingredient": ing})
    return failures


def run_efficiency_benchmark(
    n_recipes: int = 5,
    use_reranker: bool = True,
) -> dict[str, Any]:
    """Time build_cart for n_recipes; report mean latency and per-ingredient time."""
    df_recipes = load_recipes()
    sample = df_recipes.sample(n=min(n_recipes, len(df_recipes)), random_state=42)
    times = []
    n_ingredients_list = []
    for _, row in sample.iterrows():
        name = row[RECIPE_NAME_COL]
        t0 = time.perf_counter()
        cart, _, ingredients = build_cart(name, df_recipes=df_recipes, use_reranker=use_reranker)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        n_ingredients_list.append(len(ingredients))
    n_ing = sum(n_ingredients_list)
    return {
        "n_recipes": n_recipes,
        "use_reranker": use_reranker,
        "mean_latency_seconds": sum(times) / len(times) if times else 0,
        "total_seconds": sum(times),
        "mean_ingredients_per_recipe": n_ing / len(n_ingredients_list) if n_ingredients_list else 0,
        "mean_time_per_ingredient_seconds": sum(times) / n_ing if n_ing else 0,
    }


def _cart_row_matches_ingredient(cart_row: dict[str, Any], iname: str) -> bool:
    """
    True if this cart row corresponds to the given ingredient.
    Prefer matching by cart row's ingredient/ingredients (set by pipeline); fall back to product name.
    """
    iname = (iname or "").strip().lower()
    if not iname:
        return False
    # 1. Match by cart row's ingredient (single string from quantity_optimiser)
    row_ing = (cart_row.get("ingredient") or "").strip().lower()
    if row_ing and (iname in row_ing or row_ing in iname):
        return True
    # 2. Match by cart row's ingredients list (after aggregation)
    row_ings = cart_row.get("ingredients") or []
    if isinstance(row_ings, str):
        row_ings = [row_ings]
    for ri in row_ings:
        ri = (ri.strip().lower() if isinstance(ri, str) else str(ri).strip().lower())
        if ri and (iname in ri or ri in iname):
            return True
    # 3. Fall back: ingredient (or its tokens) appears in product name
    product = (cart_row.get("product") or "").lower()
    if iname in product:
        return True
    if any(t in product for t in iname.split()):
        return True
    return False


def run_quantity_verification_sample(
    n_recipes: int = 15,
    ingredients_to_check: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Run build_cart on n_recipes and collect salt/sugar (or given ingredients) and chosen product/quantity.
    For manual verification: are quantities and package sizes sensible?
    Matches cart rows by pipeline ingredient/ingredients first, then by product name.
    """
    if ingredients_to_check is None:
        ingredients_to_check = ["salt", "sugar"]
    df_recipes = load_recipes()
    sample = df_recipes.sample(n=min(n_recipes, len(df_recipes)), random_state=43)
    rows_out = []
    for _, row in sample.iterrows():
        name = row[RECIPE_NAME_COL]
        cart, _, ingredients = build_cart(name, df_recipes=df_recipes)
        for ing in ingredients:
            iname = (ing.get("ingredient") or "").strip().lower()
            if not any(check in iname for check in ingredients_to_check):
                continue
            # Find cart item: first by pipeline ingredient/ingredients, then by product name
            matched_cart = None
            for c in cart:
                if _cart_row_matches_ingredient(c, iname):
                    matched_cart = c
                    break
            rows_out.append({
                "recipe_name": name[:50],
                "ingredient": ing.get("ingredient"),
                "quantity_needed": ing.get("quantity"),
                "unit": ing.get("unit"),
                "cart_product": (matched_cart.get("product", "") or "")[:50] if matched_cart else "",
                "cart_quantity": matched_cart.get("quantity", "") if matched_cart else "",
                "cart_price": matched_cart.get("sale_price", "") if matched_cart else "",
            })
    return rows_out
