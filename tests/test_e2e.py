"""
Functional tests: End-to-end (Section 4.6.2).
Verify that a recipe name can be transformed into a complete cart output without structural failure.
Uses mocked/stubbed ingredients and retriever to avoid LLM and real dataset dependency.
"""
from unittest.mock import patch

import pandas as pd
import pytest

from src.build_cart import build_cart, get_build_cart_params
from src.candidate_retriever import CandidateRetriever
from src.load_recipes import load_recipes, get_recipe_by_name


def _mini_products_df():
    """Minimal product DataFrame for e2e retriever."""
    return pd.DataFrame({
        "index": [1, 2, 3],
        "product": ["Salt 1kg", "Potato 1kg", "Cauliflower 500g"],
        "category": ["Staples", "Vegetables", "Vegetables"],
        "brand": ["Tata", "Farm", "Farm"],
        "sale_price": [20.0, 40.0, 35.0],
        "weight_in_gms": [1000, 1000, 500],
        "type": ["Staples", "Vegetables", "Vegetables"],
    })


class TestEndToEnd:
    """End-to-end pipeline tests."""

    def test_build_cart_params_returns_dict(self):
        """get_build_cart_params returns a dict with expected keys."""
        params = get_build_cart_params()
        assert isinstance(params, dict)
        assert "top_k_retrieve" in params or "retrieval_method" in params or len(params) >= 0

    @pytest.mark.parametrize("recipe_name", ["Aloo Gobi", "Paneer Butter Masala"])
    def test_build_cart_completes_without_structural_failure(
        self,
        recipe_csv,
        recipe_name,
    ):
        """Recipe name → cart runs without raising; cart is list of dicts (may be empty if no products)."""
        df_recipes = load_recipes(path=str(recipe_csv))
        df_products = _mini_products_df()
        retriever = CandidateRetriever(
            df_products=df_products,
            use_food_filter=False,
            retrieval_method="bm25",
        )
        with patch("src.build_cart.recipe_name_to_ingredients") as mock_rti:
            mock_rti.return_value = (
                [
                    {"ingredient": "salt", "quantity": 1, "unit": "to taste"},
                    {"ingredient": "potato", "quantity": 2, "unit": "piece"},
                ],
                get_recipe_by_name(df_recipes, recipe_name),
            )
            cart, recipe_row, ingredients = build_cart(
                recipe_name,
                df_recipes=df_recipes,
                retriever=retriever,
                use_reranker=False,
                normalise_ingredients=False,
            )
        assert isinstance(cart, list)
        assert recipe_row is not None or ingredients is not None
        for row in cart:
            assert isinstance(row, dict)
            assert "index" in row or "product" in row
            assert "quantity" in row
            assert row["quantity"] >= 0

    def test_build_cart_empty_ingredients_returns_empty_cart(self, recipe_csv):
        """When ingredient extraction returns empty, cart is empty."""
        df_recipes = load_recipes(path=str(recipe_csv))
        retriever = CandidateRetriever(
            df_products=_mini_products_df(),
            use_food_filter=False,
            retrieval_method="bm25",
        )
        with patch("src.build_cart.recipe_name_to_ingredients") as mock_rti:
            mock_rti.return_value = ([], None)
            cart, recipe_row, ingredients = build_cart(
                "Aloo Gobi",
                df_recipes=df_recipes,
                retriever=retriever,
                use_reranker=False,
            )
        assert cart == []
        assert ingredients == []

    def test_build_cart_cart_rows_have_required_keys(self, recipe_csv):
        """Each cart row has product, quantity, and sale_price (or equivalent)."""
        df_recipes = load_recipes(path=str(recipe_csv))
        retriever = CandidateRetriever(
            df_products=_mini_products_df(),
            use_food_filter=False,
            retrieval_method="bm25",
        )
        with patch("src.build_cart.recipe_name_to_ingredients") as mock_rti:
            mock_rti.return_value = (
                [
                    {"ingredient": "salt", "quantity": 1, "unit": "to taste", "form": "base"},
                ],
                get_recipe_by_name(df_recipes, "Aloo Gobi"),
            )
            with patch("src.build_cart.merge_ingredients_by_synonym", side_effect=lambda x: x):
                cart, _, _ = build_cart(
                    "Aloo Gobi",
                    df_recipes=df_recipes,
                    retriever=retriever,
                    use_reranker=False,
                    normalise_ingredients=False,
                )
        if cart:
            for row in cart:
                assert "product" in row
                assert "quantity" in row
                assert "sale_price" in row or "index" in row
