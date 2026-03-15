"""
Pytest fixtures for recipe-to-cart functional tests.
Uses temporary CSVs and small DataFrames so tests do not require real dataset or LLM.
"""
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a minimal project root with config and dataset dirs for path resolution."""
    (tmp_path / "config").mkdir()
    (tmp_path / "dataset").mkdir()
    (tmp_path / "output").mkdir()
    return tmp_path


@pytest.fixture
def recipe_csv(tmp_path: Path) -> Path:
    """Minimal RecipeData.csv with required columns."""
    path = tmp_path / "RecipeData.csv"
    df = pd.DataFrame({
        "TranslatedRecipeName": ["Aloo Gobi", "Paneer Butter Masala", "Dal Fry"],
        "TranslatedIngredients": [
            "2 potatoes, 1 cup cauliflower, salt to taste",
            "200g paneer, 2 tbsp butter, 1 cup tomato puree",
            "1 cup toor dal, 1 onion, 2 tomatoes",
        ],
        "TranslatedInstructions": [
            "Chop vegetables. Cook with spices.",
            "Fry paneer. Add gravy.",
            "Cook dal. Temper with onions.",
        ],
        "Cleaned-Ingredients": [
            "potato, cauliflower, salt",
            "paneer, butter, tomato",
            "toor dal, onion, tomato",
        ],
        "Cuisine": ["North Indian", "North Indian", "North Indian"],
    })
    df.to_csv(path, index=False, encoding="utf-8")
    return path


@pytest.fixture
def bigbasket_products_csv(tmp_path: Path) -> Path:
    """Minimal BigBasket-style products CSV (unified schema: product, category)."""
    path = tmp_path / "BigBasketProducts.csv"
    df = pd.DataFrame({
        "product": ["Fresh Potato 1kg", "Cauliflower 500g", "Salt 1kg", "Paneer 200g", "Butter 100g", "Tomato 1kg"],
        "category": ["Vegetables", "Vegetables", "Staples", "Dairy", "Dairy", "Vegetables"],
        "brand": ["Farm", "Farm", "Tata", "Amul", "Amul", "Farm"],
        "sale_price": [40.0, 35.0, 20.0, 80.0, 50.0, 60.0],
        "type": ["Vegetables", "Vegetables", "Staples", "Dairy", "Dairy", "Vegetables"],
    })
    df.to_csv(path, index=False, encoding="latin-1")
    return path


@pytest.fixture
def zepto_products_csv(tmp_path: Path) -> Path:
    """Minimal Zepto-style CSV: name, Category, discountedSellingPrice (×100), mrp (×100), weightInGms."""
    path = tmp_path / "zepto_v2.csv"
    df = pd.DataFrame({
        "name": ["Potato 1kg", "Cauliflower 500g", "Salt 1kg", "Paneer 200g"],
        "Category": ["Vegetables", "Vegetables", "Staples", "Dairy"],
        "discountedSellingPrice": [4000, 3500, 2000, 8000],  # 40, 35, 20, 80 rupees
        "mrp": [4500, 4000, 2200, 8500],
        "weightInGms": [1000, 500, 1000, 200],
    })
    df.to_csv(path, index=False, encoding="utf-8")
    return path


@pytest.fixture
def df_products_mini() -> pd.DataFrame:
    """Small product DataFrame for retriever tests (unified schema with index)."""
    df = pd.DataFrame({
        "index": [1, 2, 3, 4, 5],
        "product": ["Fresh Potato 1kg", "Cauliflower 500g", "Salt 1kg", "Paneer 200g", "Butter 100g"],
        "category": ["Vegetables", "Vegetables", "Staples", "Dairy", "Dairy"],
        "brand": ["Farm", "Farm", "Tata", "Amul", "Amul"],
        "sale_price": [40.0, 35.0, 20.0, 80.0, 50.0],
        "weight_in_gms": [1000, 500, 1000, 200, 100],
        "type": ["Vegetables", "Vegetables", "Staples", "Dairy", "Dairy"],
        "description": ["Potato", "Cauliflower", "Iodised salt", "Paneer", "Butter"],
    })
    return df


@pytest.fixture
def form_rules_mini() -> dict:
    """Minimal form rules for filtering tests (maida/besan exclude, base-form preference)."""
    return {
        "derived_form_keywords": ["juice", "butter", "powder", "paste", "spread"],
        "base_form_preferred_for": ["tomato", "onion", "maida"],
        "ingredient_avoid_product_keywords": {
            "flour": {
                "ingredient_tokens": ["maida", "all-purpose flour"],
                "product_avoid": ["besan", "gram flour"],
                "exclude": True,
            },
        },
    }
