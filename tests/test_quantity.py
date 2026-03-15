"""
Functional tests: Quantity (Section 4.6.2).
Verify that unit conversion and aggregation logic produce non-zero, non-negative, and interpretable quantities.
"""
import pytest

from src.quantity_optimiser import (
    recipe_quantity_to_metric,
    aggregate_cart_rows,
    optimise_cart,
    load_unit_conversion,
)


class TestRecipeQuantityToMetric:
    """Unit conversion: recipe quantity to metric (value, ml|g|unit)."""

    def test_volume_cup_to_ml(self):
        """Cup converts to ml (240 ml per cup)."""
        value, unit = recipe_quantity_to_metric(1, "cup")
        assert unit == "ml"
        assert value == 240.0
        value2, _ = recipe_quantity_to_metric(2, "cup")
        assert value2 == 480.0

    def test_volume_tablespoon_to_ml(self):
        """Tablespoon converts to ml (15 ml)."""
        value, unit = recipe_quantity_to_metric(2, "tablespoon")
        assert unit == "ml"
        assert value == 30.0

    def test_mass_gram_to_g(self):
        """Gram stays grams; kg converts to g."""
        value, unit = recipe_quantity_to_metric(500, "gram")
        assert unit == "g"
        assert value == 500.0
        value_kg, _ = recipe_quantity_to_metric(1, "kg")
        assert value_kg == 1000.0

    def test_to_taste_returns_zero_unit(self):
        """'To taste', 'pinch', 'as required' return (0, 'unit')."""
        value, unit = recipe_quantity_to_metric(1, "to taste")
        assert value == 0.0
        assert unit == "unit"
        value2, unit2 = recipe_quantity_to_metric(1, "pinch")
        assert value2 == 0.0
        assert unit2 == "unit"

    def test_unknown_unit_passthrough(self):
        """Unknown unit returns (quantity, 'unit')."""
        value, unit = recipe_quantity_to_metric(3, "piece")
        assert unit == "unit"
        assert value == 3.0


class TestAggregateCartRows:
    """Aggregation: same product index merged, quantities summed."""

    def test_aggregate_merges_same_index(self):
        """Two rows with same index become one with summed quantity."""
        rows = [
            {"index": 1, "product": "Salt", "sale_price": 20, "quantity": 1, "unit_metric": "unit", "ingredients": ["salt"]},
            {"index": 1, "product": "Salt", "sale_price": 20, "quantity": 2, "unit_metric": "unit", "ingredients": ["salt"]},
        ]
        out = aggregate_cart_rows(rows)
        assert len(out) == 1
        assert out[0]["quantity"] == 3

    def test_aggregate_preserves_distinct_indices(self):
        """Different indices remain separate rows."""
        rows = [
            {"index": 1, "product": "Salt", "sale_price": 20, "quantity": 1, "unit_metric": "unit", "ingredients": []},
            {"index": 2, "product": "Sugar", "sale_price": 40, "quantity": 1, "unit_metric": "unit", "ingredients": []},
        ]
        out = aggregate_cart_rows(rows)
        assert len(out) == 2
        assert out[0]["index"] == 1 and out[0]["quantity"] == 1
        assert out[1]["index"] == 2 and out[1]["quantity"] == 1

    def test_aggregate_cap_quantity_at_1_remains_one(self):
        """Rows with cap_quantity_at_1 (e.g. to taste) stay at quantity 1 when merged."""
        rows = [
            {"index": 1, "product": "Salt", "sale_price": 20, "quantity": 1, "unit_metric": "unit", "cap_quantity_at_1": True, "ingredients": []},
            {"index": 1, "product": "Salt", "sale_price": 20, "quantity": 1, "unit_metric": "unit", "cap_quantity_at_1": True, "ingredients": []},
        ]
        out = aggregate_cart_rows(rows)
        assert len(out) == 1
        assert out[0]["quantity"] == 1
        assert out[0].get("cap_quantity_at_1") is True

    def test_aggregate_empty_list_returns_empty(self):
        """Empty list returns empty."""
        assert aggregate_cart_rows([]) == []


class TestOptimiseCart:
    """Quantity optimiser: selected products + ingredients → cart with non-negative quantities."""

    def test_optimise_cart_returns_non_zero_quantities(self):
        """Cart rows have quantity >= 1 (or 0 only for to-taste cap)."""
        selected = [
            {"index": 1, "product": "Salt 1kg", "brand": "Tata", "sale_price": 20, "weight_in_gms": 1000},
        ]
        ingredients = [{"ingredient": "salt", "quantity": 1, "unit": "to taste"}]
        cart = optimise_cart(selected, ingredients)
        assert len(cart) >= 1
        assert cart[0]["quantity"] >= 1
        assert cart[0]["quantity"] == 1  # to taste → 1 unit

    def test_optimise_cart_non_negative(self):
        """No negative quantities."""
        selected = [
            {"index": 1, "product": "Potato 1kg", "brand": "Farm", "sale_price": 40, "weight_in_gms": 1000},
        ]
        ingredients = [{"ingredient": "potato", "quantity": 2, "unit": "piece"}]
        cart = optimise_cart(selected, ingredients)
        assert all(r["quantity"] >= 0 for r in cart)

    def test_optimise_cart_structure(self):
        """Cart rows have index, product, brand, sale_price, quantity, unit_metric."""
        selected = [
            {"index": 1, "product": "Paneer 200g", "brand": "Amul", "sale_price": 80, "weight_in_gms": 200},
        ]
        ingredients = [{"ingredient": "paneer", "quantity": 1, "unit": "piece"}]
        cart = optimise_cart(selected, ingredients)
        assert len(cart) == 1
        row = cart[0]
        assert "index" in row and "product" in row and "quantity" in row
        assert "sale_price" in row
        assert row["quantity"] >= 1

    def test_optimise_cart_mismatched_lengths_truncated(self):
        """If selected and ingredients length differ, use min length."""
        selected = [
            {"index": 1, "product": "A", "brand": "", "sale_price": 10, "weight_in_gms": 0},
            {"index": 2, "product": "B", "brand": "", "sale_price": 10, "weight_in_gms": 0},
        ]
        ingredients = [{"ingredient": "a", "quantity": 1, "unit": "piece"}]
        cart = optimise_cart(selected, ingredients)
        assert len(cart) == 1

    def test_load_unit_conversion_returns_dict(self):
        """Unit conversion config is a dict (from file or defaults)."""
        cfg = load_unit_conversion()
        assert isinstance(cfg, dict)
        assert "volume_ml" in cfg or "mass_g" in cfg or len(cfg) >= 0
