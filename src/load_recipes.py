"""
Load and validate recipe data from RecipeData.csv.
Preparation and data.
"""
from pathlib import Path
from typing import Optional

import pandas as pd


# Default path relative to project root
DEFAULT_RECIPE_PATH = "dataset/RecipeData.csv"
# Column used for pre-parsed ingredient list
CLEANED_INGREDIENTS_COL = "Cleaned-Ingredients"
RECIPE_NAME_COL = "TranslatedRecipeName"


def get_project_root() -> Path:
    """Project root: directory containing 'dataset' and 'config'."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "dataset").is_dir() and (current / "config").is_dir():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


def load_recipes(
    path: Optional[str] = None,
    encoding: str = "utf-8",
    on_bad_lines: str = "warn",
) -> pd.DataFrame:
    """
    Load RecipeData.csv with robust CSV parsing.

    Parameters
    ----------
    path : str or None
        Path to RecipeData.csv. If None, uses config/paths.yaml or default.
    encoding : str
        File encoding (default utf-8; fallback to latin-1 if needed).
    on_bad_lines : str
        Passed to pandas read_csv: 'warn', 'skip', or 'error'.

    Returns
    -------
    pd.DataFrame
        Recipe table with columns including TranslatedRecipeName,
        TranslatedIngredients, Cleaned-Ingredients, Cuisine, etc.
    """
    if path is None:
        root = get_project_root()
        config_path = root / "config" / "paths.yaml"
        if config_path.exists():
            try:
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                path = config.get("recipe_data", DEFAULT_RECIPE_PATH)
            except Exception:
                path = DEFAULT_RECIPE_PATH
        else:
            path = DEFAULT_RECIPE_PATH
        path = root / path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Recipe data not found: {path}")

    try:
        df = pd.read_csv(path, encoding=encoding, on_bad_lines=on_bad_lines)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines=on_bad_lines)

    # Basic validation
    if RECIPE_NAME_COL not in df.columns:
        raise ValueError(f"Expected column '{RECIPE_NAME_COL}' in recipe data. Columns: {list(df.columns)}")
    if CLEANED_INGREDIENTS_COL not in df.columns:
        raise ValueError(f"Expected column '{CLEANED_INGREDIENTS_COL}'. Columns: {list(df.columns)}")

    return df


def parse_cleaned_ingredients(ingredient_string: str) -> list[str]:
    """
    Parse Cleaned-Ingredients string into a list of ingredient names.

    Parameters
    ----------
    ingredient_string : str
        Comma-separated ingredient names (e.g. from Cleaned-Ingredients).

    Returns
    -------
    list[str]
        Stripped, non-empty ingredient names.
    """
    if pd.isna(ingredient_string) or not isinstance(ingredient_string, str):
        return []
    return [s.strip() for s in ingredient_string.split(",") if s.strip()]


def get_recipe_by_name(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """
    Return the first recipe row whose TranslatedRecipeName matches (case-insensitive).

    Parameters
    ----------
    df : pd.DataFrame
        Loaded recipe dataframe.
    name : str
        Recipe name to search.

    Returns
    -------
    pd.Series or None
        First matching row or None.
    """
    name_clean = name.strip().lower()
    if RECIPE_NAME_COL not in df.columns:
        return None
    mask = df[RECIPE_NAME_COL].astype(str).str.lower() == name_clean
    if mask.any():
        return df.loc[mask].iloc[0]
    return None


TRANSLATED_INGREDIENTS_COL = "TranslatedIngredients"
TRANSLATED_INSTRUCTIONS_COL = "TranslatedInstructions"


def get_recipe_text_for_llm(
    recipe_row: pd.Series,
    include_instructions: bool = True,
) -> str:
    """
    Build a single text blob from a recipe row for LLM ingredient extraction.

    Parameters
    ----------
    recipe_row : pd.Series
        One row from the recipe dataframe (e.g. from get_recipe_by_name).
    include_instructions : bool
        If True, append TranslatedInstructions after ingredients.

    Returns
    -------
    str
        Text suitable for the ingredient-extractor prompt (ingredients list,
        optionally followed by instructions).
    """
    parts = []
    if TRANSLATED_INGREDIENTS_COL in recipe_row.index and pd.notna(recipe_row.get(TRANSLATED_INGREDIENTS_COL)):
        parts.append(str(recipe_row[TRANSLATED_INGREDIENTS_COL]).strip())
    if include_instructions and TRANSLATED_INSTRUCTIONS_COL in recipe_row.index and pd.notna(recipe_row.get(TRANSLATED_INSTRUCTIONS_COL)):
        parts.append(str(recipe_row[TRANSLATED_INSTRUCTIONS_COL]).strip())
    return "\n\n".join(parts) if parts else ""


if __name__ == "__main__":
    df = load_recipes()
    print("Recipe data shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Sample recipe name:", df[RECIPE_NAME_COL].iloc[0])
    sample_ing = df[CLEANED_INGREDIENTS_COL].iloc[0]
    print("Sample cleaned ingredients (parsed):", parse_cleaned_ingredients(sample_ing)[:5])
