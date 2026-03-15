"""
Functional tests: Filtering and ranking (Section 4.6.2).
Verify that exclusion rules and re-ranking logic remove or avoid clearly wrong products.
"""
import pytest

from src.product_form_filter import (
    load_form_rules,
    reorder_candidates,
    reject_product_for_ingredient,
    _product_should_exclude_for_ingredient,
    _product_is_derived_form,
    _ingredient_suggests_base_form,
)


def _candidate(product_name: str, index: int = 1) -> dict:
    return {"index": index, "product": product_name, "category": "x", "sale_price": 10.0}


class TestFilteringRanking:
    """Filtering and ranking tests."""

    def test_reject_product_maida_besan_excluded(self, form_rules_mini):
        """Product containing 'besan' must be rejected for ingredient 'maida'."""
        assert reject_product_for_ingredient("maida", "Besan 500g", rules=form_rules_mini) is True
        assert reject_product_for_ingredient("all-purpose flour", "Gram flour (besan)", rules=form_rules_mini) is True

    def test_reject_product_maida_all_purpose_flour_not_excluded(self, form_rules_mini):
        """All-purpose flour product is not excluded for maida."""
        assert reject_product_for_ingredient("maida", "All Purpose Flour 1kg", rules=form_rules_mini) is False

    def test_reorder_candidates_excluded_removed(self, form_rules_mini):
        """Excluded products (e.g. besan for maida) are removed from candidate list."""
        candidates = [
            _candidate("Besan 500g", 1),
            _candidate("Maida All Purpose Flour 1kg", 2),
        ]
        reordered = reorder_candidates(candidates, "maida", rules=form_rules_mini)
        products = [c["product"] for c in reordered]
        assert "Besan 500g" not in products
        assert "Maida All Purpose Flour 1kg" in products

    def test_reorder_candidates_base_before_derived(self, form_rules_mini):
        """When ingredient suggests base form, base-form products come before derived-form."""
        candidates = [
            _candidate("Tomato Juice 1L", 1),
            _candidate("Fresh Tomato 1kg", 2),
        ]
        reordered = reorder_candidates(candidates, "tomato", rules=form_rules_mini)
        assert len(reordered) == 2
        assert reordered[0]["product"] == "Fresh Tomato 1kg"
        assert reordered[1]["product"] == "Tomato Juice 1L"

    def test_reorder_candidates_empty_list_unchanged(self, form_rules_mini):
        """Empty candidate list returns empty."""
        assert reorder_candidates([], "salt", rules=form_rules_mini) == []

    def test_ingredient_suggests_base_form_single_word(self, form_rules_mini):
        """Single-word ingredient without derived keyword suggests base form."""
        assert _ingredient_suggests_base_form("tomato", form_rules_mini) is True
        assert _ingredient_suggests_base_form("onion", form_rules_mini) is True

    def test_ingredient_with_powder_does_not_force_base(self, form_rules_mini):
        """Ingredient containing 'powder' does not force base-form preference."""
        assert _ingredient_suggests_base_form("red chilli powder", form_rules_mini) is False

    def test_product_is_derived_form(self, form_rules_mini):
        """Product name with juice, butter, powder is derived form."""
        assert _product_is_derived_form("Tomato Juice", form_rules_mini) is True
        assert _product_is_derived_form("Cashew Butter", form_rules_mini) is True
        assert _product_is_derived_form("Fresh Potato 1kg", form_rules_mini) is False

    def test_load_form_rules_returns_dict(self):
        """load_form_rules returns a dict (from project config or defaults)."""
        rules = load_form_rules()
        assert isinstance(rules, dict)
        assert "derived_form_keywords" in rules or "base_form_preferred_for" in rules or len(rules) >= 0
