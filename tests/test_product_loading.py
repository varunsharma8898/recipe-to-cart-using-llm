"""
Functional tests: Product loading (Section 4.6.2).
Verify that BigBasket and Zepto can both be loaded into the unified internal schema.
"""
import pandas as pd
import pytest

from src.load_products import (
    load_products,
    load_zepto_products,
    filter_food_products,
    load_food_categories,
    CATEGORY_COL,
)


# Zepto column mapping for tests (no dependency on project config)
ZEPTO_COLUMN_CONFIG = {
    "zepto_columns": {
        "name": "product",
        "Category": "category",
        "discountedSellingPrice": "sale_price",
        "mrp": "market_price",
        "weightInGms": "weight_in_gms",
    },
    "defaults": {"brand": "", "description": "", "type": "", "rating": None},
}


class TestProductLoading:
    """Product loading tests."""

    def test_load_bigbasket_from_path(self, bigbasket_products_csv):
        """BigBasket CSV loads into DataFrame with product and category."""
        df = load_products(path=str(bigbasket_products_csv), source="bigbasket")
        assert isinstance(df, pd.DataFrame)
        assert "product" in df.columns
        assert CATEGORY_COL in df.columns
        assert len(df) == 6

    def test_load_bigbasket_unified_schema_columns(self, bigbasket_products_csv):
        """Unified schema includes expected columns."""
        df = load_products(path=str(bigbasket_products_csv), source="bigbasket")
        assert "product" in df.columns
        assert CATEGORY_COL in df.columns
        # sale_price may come from CSV or be missing
        assert df["product"].dtype == object or df["product"].notna().all()

    def test_load_zepto_from_path(self, zepto_products_csv):
        """Zepto CSV loads into unified schema with normalized prices and weight."""
        df = load_zepto_products(
            path=str(zepto_products_csv),
            column_config=ZEPTO_COLUMN_CONFIG,
        )
        assert "product" in df.columns
        assert CATEGORY_COL in df.columns
        assert "sale_price" in df.columns
        assert "weight_in_gms" in df.columns
        assert len(df) == 4

    def test_load_zepto_price_divided_by_100(self, zepto_products_csv):
        """Zepto prices (×100 in CSV) are converted to rupee values."""
        df = load_zepto_products(
            path=str(zepto_products_csv),
            column_config=ZEPTO_COLUMN_CONFIG,
        )
        # discountedSellingPrice 4000 -> 40.0
        assert df["sale_price"].iloc[0] == 40.0
        assert df["market_price"].iloc[0] == 45.0

    def test_load_zepto_weight_in_gms_present(self, zepto_products_csv):
        """Zepto weightInGms becomes weight_in_gms in unified schema."""
        df = load_zepto_products(
            path=str(zepto_products_csv),
            column_config=ZEPTO_COLUMN_CONFIG,
        )
        assert df["weight_in_gms"].iloc[0] == 1000
        assert df["weight_in_gms"].iloc[3] == 200

    def test_load_zepto_index_column_added(self, zepto_products_csv):
        """Loaded Zepto DataFrame has integer index column for retriever/cart."""
        df = load_zepto_products(
            path=str(zepto_products_csv),
            column_config=ZEPTO_COLUMN_CONFIG,
        )
        assert "index" in df.columns
        assert list(df["index"]) == [1, 2, 3, 4]

    def test_load_products_missing_file_raises(self, tmp_path):
        """Missing CSV path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_products(path=str(tmp_path / "nonexistent.csv"), source="bigbasket")

    def test_load_products_missing_required_columns_raises(self, tmp_path):
        """CSV without product or category raises ValueError."""
        path = tmp_path / "bad.csv"
        pd.DataFrame({"other": [1]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="product|category"):
            load_products(path=str(path), source="bigbasket")

    def test_filter_food_products_subset(self, bigbasket_products_csv):
        """filter_food_products returns only rows in given category set."""
        df = load_products(path=str(bigbasket_products_csv), source="bigbasket")
        food_cats = {"Vegetables", "Staples"}
        filtered = filter_food_products(df, food_categories=food_cats)
        assert len(filtered) < len(df)
        assert set(filtered[CATEGORY_COL].unique()).issubset(food_cats)

    def test_filter_food_products_empty_set_returns_copy(self, bigbasket_products_csv):
        """Empty food_categories returns full copy."""
        df = load_products(path=str(bigbasket_products_csv), source="bigbasket")
        out = filter_food_products(df, food_categories=set())
        assert len(out) == len(df)
        assert out is not df
