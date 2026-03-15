"""
Functional tests: LLM output parsing (Section 4.6.2).
Verify that ingredient extractor and normaliser outputs are converted into valid structured objects.
"""
import pandas as pd
import pytest

from src.ingredient_extractor import (
    _parse_llm_json,
    fallback_from_cleaned_ingredients,
)
from src.load_recipes import CLEANED_INGREDIENTS_COL
from src.quantity_normaliser import _parse_normaliser_json


class TestIngredientExtractorParsing:
    """Ingredient extractor LLM output parsing."""

    def test_parse_valid_json_array(self):
        """Valid JSON array of objects with ingredient, quantity, unit is parsed."""
        raw = '[{"ingredient": "butter", "quantity": 2, "unit": "tablespoons"}]'
        out = _parse_llm_json(raw)
        assert len(out) == 1
        assert out[0]["ingredient"] == "butter"
        assert out[0]["quantity"] == 2
        assert out[0]["unit"] == "tablespoons"

    def test_parse_wrapped_in_markdown_code_block(self):
        """JSON wrapped in ```json ... ``` is extracted."""
        raw = '```json\n[{"ingredient": "salt", "quantity": 1, "unit": "to taste"}]\n```'
        out = _parse_llm_json(raw)
        assert len(out) == 1
        assert out[0]["ingredient"] == "salt"
        assert out[0]["unit"] == "to taste"

    def test_parse_accepts_name_alias(self):
        """Object with 'name' instead of 'ingredient' is accepted."""
        raw = '[{"name": "maida", "quantity": 1, "unit": "cup"}]'
        out = _parse_llm_json(raw)
        assert len(out) == 1
        assert out[0]["ingredient"] == "maida"

    def test_parse_missing_quantity_defaults_to_one(self):
        """Missing quantity defaults to 1."""
        raw = '[{"ingredient": "oil", "unit": "tablespoon"}]'
        out = _parse_llm_json(raw)
        assert out[0]["quantity"] == 1

    def test_parse_missing_unit_defaults_to_piece(self):
        """Missing unit defaults to 'piece'."""
        raw = '[{"ingredient": "onion", "quantity": 2}]'
        out = _parse_llm_json(raw)
        assert out[0]["unit"] == "piece"

    def test_parse_invalid_json_returns_empty_list(self):
        """Invalid JSON returns empty list."""
        assert _parse_llm_json("not json") == []
        assert _parse_llm_json("") == []
        assert _parse_llm_json("[]") == []

    def test_parse_non_list_returns_empty_list(self):
        """Top-level non-array returns empty list."""
        assert _parse_llm_json('{"ingredient": "x"}') == []

    def test_parse_skips_items_without_ingredient_or_name(self):
        """Items without ingredient/name are skipped."""
        raw = '[{"quantity": 1, "unit": "cup"}, {"ingredient": "flour", "quantity": 1, "unit": "cup"}]'
        out = _parse_llm_json(raw)
        assert len(out) == 1
        assert out[0]["ingredient"] == "flour"

    def test_fallback_from_cleaned_ingredients(self):
        """Fallback builds structured list from Cleaned-Ingredients with quantity=1, unit=piece."""
        row = pd.Series({CLEANED_INGREDIENTS_COL: "potato, cauliflower, salt"})
        out = fallback_from_cleaned_ingredients(row)
        assert len(out) == 3
        assert out[0]["ingredient"] == "potato" and out[0]["quantity"] == 1 and out[0]["unit"] == "piece"
        assert out[1]["ingredient"] == "cauliflower"
        assert out[2]["ingredient"] == "salt"

    def test_fallback_empty_cleaned_returns_empty_list(self):
        """Empty or NaN Cleaned-Ingredients returns empty list."""
        row = pd.Series({CLEANED_INGREDIENTS_COL: ""})
        assert fallback_from_cleaned_ingredients(row) == []
        row2 = pd.Series({CLEANED_INGREDIENTS_COL: pd.NA})
        assert fallback_from_cleaned_ingredients(row2) == []


class TestQuantityNormaliserParsing:
    """Quantity normaliser LLM output parsing."""

    def test_parse_normaliser_valid_json(self):
        """Valid normaliser JSON with ingredient, quantity, unit, form is parsed."""
        raw = '[{"ingredient": "all-purpose flour", "quantity": 1, "unit": "cup", "form": "base"}]'
        out = _parse_normaliser_json(raw)
        assert len(out) == 1
        assert out[0]["ingredient"] == "all-purpose flour"
        assert out[0]["quantity"] == 1
        assert out[0]["unit"] == "cup"
        assert out[0]["form"] == "base"

    def test_parse_normaliser_form_defaults_to_base(self):
        """Missing form defaults to 'base'."""
        raw = '[{"ingredient": "salt", "quantity": 1, "unit": "to taste"}]'
        out = _parse_normaliser_json(raw)
        assert out[0]["form"] == "base"

    def test_parse_normaliser_accepts_valid_forms(self):
        """Form can be base, powder, juice, butter, paste, leaves, oil."""
        raw = '[{"ingredient": "turmeric", "quantity": 1, "unit": "teaspoon", "form": "powder"}]'
        out = _parse_normaliser_json(raw)
        assert out[0]["form"] == "powder"

    def test_parse_normaliser_invalid_form_reverted_to_base(self):
        """Invalid form value is set to 'base'."""
        raw = '[{"ingredient": "x", "quantity": 1, "unit": "piece", "form": "invalid"}]'
        out = _parse_normaliser_json(raw)
        assert out[0]["form"] == "base"

    def test_parse_normaliser_wrapped_in_markdown(self):
        """Markdown code block is stripped."""
        raw = '```json\n[{"ingredient": "oil", "quantity": 2, "unit": "tablespoon", "form": "oil"}]\n```'
        out = _parse_normaliser_json(raw)
        assert len(out) == 1
        assert out[0]["form"] == "oil"

    def test_parse_normaliser_invalid_json_returns_empty_list(self):
        """Invalid JSON returns empty list."""
        assert _parse_normaliser_json("not json") == []
        assert _parse_normaliser_json("") == []
