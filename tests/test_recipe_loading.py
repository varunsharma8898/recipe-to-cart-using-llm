"""
Functional tests: Recipe loading (Section 4.6.2).
Verify that recipe rows can be located by name and converted into LLM-ready text.
"""
import pandas as pd
import pytest

from src.load_recipes import (
    RECIPE_NAME_COL,
    CLEANED_INGREDIENTS_COL,
    load_recipes,
    get_recipe_by_name,
    get_recipe_text_for_llm,
    parse_cleaned_ingredients,
    TRANSLATED_INGREDIENTS_COL,
    TRANSLATED_INSTRUCTIONS_COL,
)


class TestRecipeLoading:
    """Recipe loading tests."""

    def test_load_recipes_from_path(self, recipe_csv):
        """Load recipe CSV from explicit path returns DataFrame with required columns."""
        df = load_recipes(path=str(recipe_csv))
        assert isinstance(df, pd.DataFrame)
        assert RECIPE_NAME_COL in df.columns
        assert CLEANED_INGREDIENTS_COL in df.columns
        assert len(df) == 3

    def test_load_recipes_missing_recipe_name_column_fails(self, tmp_path):
        """Missing TranslatedRecipeName column raises ValueError."""
        path = tmp_path / "bad.csv"
        pd.DataFrame({"Cleaned-Ingredients": ["a"]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="TranslatedRecipeName"):
            load_recipes(path=str(path))

    def test_load_recipes_missing_cleaned_ingredients_fails(self, tmp_path):
        """Missing Cleaned-Ingredients column raises ValueError."""
        path = tmp_path / "bad.csv"
        pd.DataFrame({"TranslatedRecipeName": ["x"]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="Cleaned-Ingredients"):
            load_recipes(path=str(path))

    def test_get_recipe_by_name_found(self, recipe_csv):
        """Recipe row is located by name (case-insensitive)."""
        df = load_recipes(path=str(recipe_csv))
        row = get_recipe_by_name(df, "Aloo Gobi")
        assert row is not None
        assert str(row[RECIPE_NAME_COL]).lower() == "aloo gobi"

    def test_get_recipe_by_name_case_insensitive(self, recipe_csv):
        """Matching is case-insensitive."""
        df = load_recipes(path=str(recipe_csv))
        row = get_recipe_by_name(df, "PANEER BUTTER MASALA")
        assert row is not None
        assert "Paneer" in str(row[RECIPE_NAME_COL])

    def test_get_recipe_by_name_not_found_returns_none(self, recipe_csv):
        """Unknown recipe name returns None."""
        df = load_recipes(path=str(recipe_csv))
        assert get_recipe_by_name(df, "Nonexistent Recipe") is None

    def test_get_recipe_text_for_llm_includes_ingredients_and_instructions(self, recipe_csv):
        """LLM-ready text combines translated ingredients and instructions."""
        df = load_recipes(path=str(recipe_csv))
        row = get_recipe_by_name(df, "Aloo Gobi")
        text = get_recipe_text_for_llm(row, include_instructions=True)
        assert "potato" in text.lower() or "Potato" in text
        assert "cauliflower" in text.lower() or "Cauliflower" in text
        assert "Chop" in text or "Cook" in text

    def test_get_recipe_text_for_llm_instructions_optional(self, recipe_csv):
        """Without instructions, only ingredients are included."""
        df = load_recipes(path=str(recipe_csv))
        row = get_recipe_by_name(df, "Aloo Gobi")
        text_without = get_recipe_text_for_llm(row, include_instructions=False)
        text_with = get_recipe_text_for_llm(row, include_instructions=True)
        assert len(text_with) >= len(text_without)
        assert "potato" in text_without.lower() or "Potato" in text_without

    def test_parse_cleaned_ingredients_returns_list(self):
        """Parse cleaned ingredients string into list of names."""
        out = parse_cleaned_ingredients("potato, cauliflower, salt")
        assert out == ["potato", "cauliflower", "salt"]

    def test_parse_cleaned_ingredients_strips_and_drops_empty(self):
        """Whitespace is stripped; empty segments dropped."""
        out = parse_cleaned_ingredients("  a ,  b , , c  ")
        assert out == ["a", "b", "c"]

    def test_parse_cleaned_ingredients_empty_or_nan_returns_empty_list(self):
        """Empty string or NaN returns empty list."""
        assert parse_cleaned_ingredients("") == []
        assert parse_cleaned_ingredients(None) == []
        assert parse_cleaned_ingredients(pd.NA) == []
