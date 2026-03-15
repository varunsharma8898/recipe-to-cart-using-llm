"""
Build cart: recipe name → ingredients → retrieve (BM25 or LLM) → re-rank → quantity optimise → cart.
"""
import sys
from typing import Any, Optional

import pandas as pd

from .load_recipes import load_recipes, RECIPE_NAME_COL
from .load_products import get_project_root
from .recipe_to_ingredients import recipe_name_to_ingredients
from .candidate_retriever import CandidateRetriever, load_food_products_cached
from .product_form_filter import reorder_candidates, reject_product_for_ingredient
from .reranker import rerank
from .quantity_optimiser import optimise_cart, aggregate_cart_rows
from .ingredient_synonyms import merge_ingredients_by_synonym


def get_build_cart_params() -> dict[str, Any]:
    """Load build_cart section from config/params.yaml with defaults."""
    defaults = {
        "top_k_retrieve": 10,
        "max_candidates_rerank": 5,
        "use_reranker": True,
        "retrieval_method": "bm25",
        "llm_retrieval_pool_size": 30,
    }
    root = get_project_root()
    path = root / "config" / "params.yaml"
    if not path.exists():
        return defaults
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        section = data.get("build_cart") or {}
        return {**defaults, **{k: v for k, v in section.items() if v is not None}}
    except Exception:
        return defaults


def build_cart(
    recipe_name: str,
    df_recipes: Optional[pd.DataFrame] = None,
    retriever: Optional[CandidateRetriever] = None,
    *,
    top_k_retrieve: int = 10,
    max_candidates_rerank: int = 5,
    use_reranker: bool = True,
    retrieval_method: Optional[str] = None,
    llm_retrieval_pool_size: Optional[int] = None,
    include_instructions: bool = True,
    normalise_ingredients: bool = True,
) -> tuple[list[dict[str, Any]], Optional[pd.Series], list[dict[str, Any]]]:
    """
    End-to-end: recipe name → cart (list of products with quantity and price).

    Parameters
    ----------
    recipe_name : str
        Recipe name (match to TranslatedRecipeName).
    df_recipes : pd.DataFrame or None
        Recipe dataframe; if None, load_recipes().
    retriever : CandidateRetriever or None
        If None, built from food products (cached or load+filter).
    top_k_retrieve : int
        Top-k candidates per ingredient (from BM25 or LLM retrieval).
    max_candidates_rerank : int
        How many candidates to send to LLM re-ranker.
    use_reranker : bool
        If True, call LLM to pick best product; else use top-1 from retrieval.
    retrieval_method : str or None
        "bm25" (default) or "llm". If None, read from config/params.yaml.
    llm_retrieval_pool_size : int or None
        When retrieval_method is "llm", BM25 pool size before LLM selects top-k. If None, from config.
    include_instructions : bool
        Passed to recipe_to_ingredients.
    normalise_ingredients : bool
        Passed to recipe_to_ingredients.

    Returns
    -------
    cart : list[dict]
        [{"index", "product", "brand", "sale_price", "quantity", "unit_metric"}, ...].
    recipe_row : pd.Series or None
        Matched recipe row.
    ingredients : list[dict]
        Structured ingredients from previous phase (for debugging).
    """
    params = get_build_cart_params()
    if retrieval_method is None:
        retrieval_method = str(params.get("retrieval_method", "bm25")).strip().lower()
    if llm_retrieval_pool_size is None:
        llm_retrieval_pool_size = int(params.get("llm_retrieval_pool_size", 30))
    if df_recipes is None:
        df_recipes = load_recipes()
    print("  [1/4] Extracting ingredients (LLM)...", flush=True)
    ingredients, recipe_row = recipe_name_to_ingredients(
        recipe_name,
        df_recipes=df_recipes,
        include_instructions=include_instructions,
        normalise=normalise_ingredients,
        use_fallback_on_failure=True,
    )
    if not ingredients:
        return [], recipe_row, ingredients
    print(f"  [2/4] Got {len(ingredients)} ingredients; merging synonyms...", flush=True)
    # Merge synonym ingredients (e.g. besan + gram flour) so we get one cart line per product
    ingredients = merge_ingredients_by_synonym(ingredients)
    n_ing = len(ingredients)

    if retriever is None:
        print("  [3/4] Loading product catalogue & building retriever...", flush=True)
        df_food = load_food_products_cached()
        # Normalize: only "llm" enables LLM retrieval; anything else is BM25
        retrieval_method = (retrieval_method or "bm25").strip().lower()
        if retrieval_method != "llm":
            retrieval_method = "bm25"
        retriever = CandidateRetriever(
            df_products=df_food,
            use_food_filter=False,
            retrieval_method=retrieval_method,
            llm_retrieval_pool_size=llm_retrieval_pool_size,
        )
        print(f"       Retrieval: {retrieval_method}" + (f" (LLM pool={llm_retrieval_pool_size})" if retrieval_method == "llm" else ""), flush=True)
    print(f"  [4/4] Retrieving & re-ranking for {n_ing} ingredients...", flush=True)

    recipe_name_ctx = recipe_row[RECIPE_NAME_COL] if recipe_row is not None else recipe_name
    selected = []
    ingredients_matched = []
    for idx, ing in enumerate(ingredients):
        name = ing.get("ingredient", "")
        print(f"      [{idx + 1}/{n_ing}] {name[:40]}{'...' if len(name) > 40 else ''}", flush=True)
        q = ing.get("quantity", 1)
        u = ing.get("unit", "piece")
        form_hint = ing.get("form", "base")
        candidates = retriever.retrieve(name, k=top_k_retrieve, recipe_name=recipe_name_ctx)
        if not candidates:
            continue
        candidates = reorder_candidates(candidates, name)
        if use_reranker and len(candidates) > 1:
            try:
                best = rerank(
                    ingredient=name,
                    quantity=float(q) if isinstance(q, (int, float)) else 1,
                    unit=str(u),
                    recipe_name=recipe_name_ctx,
                    candidates=candidates,
                    max_candidates=max_candidates_rerank,
                    form_hint=form_hint,
                )
            except Exception:
                best = candidates[0]
        else:
            best = candidates[0]
        # Post-check: if reranker returned a product that must not be used for this ingredient (e.g. besan for maida), use next valid candidate
        if best and reject_product_for_ingredient(name, best.get("product", "")):
            best = next(
                (c for c in candidates if not reject_product_for_ingredient(name, c.get("product", ""))),
                None,
            )
        if best:
            selected.append(best)
            ingredients_matched.append(ing)
    cart = optimise_cart(selected, ingredients_matched) if selected else []
    return cart, recipe_row, ingredients


def build_cart_for_recipes(
    recipe_names: list[str],
    df_recipes: Optional[pd.DataFrame] = None,
    retriever: Optional[CandidateRetriever] = None,
    *,
    top_k_retrieve: int = 10,
    max_candidates_rerank: int = 5,
    use_reranker: bool = True,
    retrieval_method: Optional[str] = None,
    llm_retrieval_pool_size: Optional[int] = None,
    include_instructions: bool = True,
    normalise_ingredients: bool = True,
) -> tuple[list[dict[str, Any]], list[tuple[Optional[pd.Series], list[dict[str, Any]]]]]:
    """
    Build a single aggregated cart from multiple recipes.
    Same product across recipes is merged into one line with combined quantity.

    Returns
    -------
    cart : list[dict]
        Single cart with merged quantities (same product index → summed quantity).
    per_recipe : list[(recipe_row, ingredients)]
        One entry per recipe for reference.
    """
    if not recipe_names:
        return [], []
    if df_recipes is None:
        df_recipes = load_recipes()
    if retriever is None:
        params = get_build_cart_params()
        rm = retrieval_method if retrieval_method is not None else str(params.get("retrieval_method", "bm25")).strip().lower()
        pool = llm_retrieval_pool_size if llm_retrieval_pool_size is not None else int(params.get("llm_retrieval_pool_size", 30))
        df_food = load_food_products_cached()
        retriever = CandidateRetriever(
            df_products=df_food,
            use_food_filter=False,
            retrieval_method=rm,
            llm_retrieval_pool_size=pool,
        )

    all_cart_rows = []
    per_recipe = []
    n_recipes = len(recipe_names)
    for r_idx, name in enumerate(recipe_names):
        print(f"  Recipe {r_idx + 1}/{n_recipes}: {name[:50]}{'...' if len(name) > 50 else ''}", flush=True)
        cart, recipe_row, ingredients = build_cart(
            name,
            df_recipes=df_recipes,
            retriever=retriever,
            top_k_retrieve=top_k_retrieve,
            max_candidates_rerank=max_candidates_rerank,
            use_reranker=use_reranker,
            retrieval_method=retrieval_method,
            llm_retrieval_pool_size=llm_retrieval_pool_size,
            include_instructions=include_instructions,
            normalise_ingredients=normalise_ingredients,
        )
        all_cart_rows.extend(cart)
        per_recipe.append((recipe_row, ingredients))
    single_cart = aggregate_cart_rows(all_cart_rows) if all_cart_rows else []
    return single_cart, per_recipe


def cart_to_rows(cart: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return cart as list of dicts with total_price and original ingredient name(s). Drops internal keys like cap_quantity_at_1."""
    out = []
    for row in cart:
        q = row.get("quantity", 1)
        p = float(row.get("sale_price", 0))
        clean = {k: v for k, v in row.items() if k not in ("cap_quantity_at_1", "need_grams", "pack_grams", "need_ml", "ml_per_unit")}
        clean["total_price"] = round(q * p, 2)
        # Expose original ingredient name(s): list always, single string when one
        ingredients = clean.get("ingredients", [])
        if len(ingredients) == 1:
            clean["ingredient"] = ingredients[0]
        out.append(clean)
    return out
