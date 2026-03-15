"""
Load and filter product catalogue (BigBasket or Zepto).
Preparation and data.
Zepto: discountedSellingPrice and mrp are ×100; loader divides by 100. weightInGms → weight_in_gms.
"""
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml


DEFAULT_PRODUCTS_PATH = "dataset/BigBasketProducts.csv"
DEFAULT_ZEPTO_PATH = "dataset/zepto/zepto_v2.csv"
CATEGORY_COL = "category"


def get_project_root() -> Path:
    """Project root: directory containing 'dataset' and 'config'."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "dataset").is_dir() and (current / "config").is_dir():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


def load_food_categories(path: Optional[Path] = None, source: Optional[str] = None) -> set[str]:
    """
    Load food category names. For Zepto use config/food_categories_zepto.txt
    (Zepto category names differ from BigBasket). Lines starting with # and empty lines are ignored.
    """
    if path is None:
        root = get_project_root()
        src = (source or _get_products_source()).strip().lower()
        filename = "food_categories_zepto.txt" if src == "zepto" else "food_categories.txt"
        path = root / "config" / filename
    path = Path(path)
    if not path.exists():
        return set()
    categories = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                categories.add(line)
    return categories


def _get_products_source() -> str:
    """Read products_source from config; default 'bigbasket'."""
    root = get_project_root()
    config_path = root / "config" / "paths.yaml"
    if not config_path.exists():
        return "bigbasket"
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        return (config.get("products_source") or "bigbasket").strip().lower()
    except Exception:
        return "bigbasket"


def _load_zepto_column_config(path: Optional[Path] = None) -> dict[str, Any]:
    if path is None:
        path = get_project_root() / "config" / "zepto_column_mapping.yaml"
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_zepto_products(
    path: Optional[str] = None,
    encoding: str = "utf-8",
    on_bad_lines: str = "warn",
    column_config: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Load Zepto CSV and normalize to internal schema.
    discountedSellingPrice and mrp are divided by 100. weightInGms → weight_in_gms.
    """
    root = get_project_root()
    if path is None:
        config_path = root / "config" / "paths.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = yaml.safe_load(f) or {}
                path = cfg.get("zepto_products", DEFAULT_ZEPTO_PATH)
            except Exception:
                path = DEFAULT_ZEPTO_PATH
        else:
            path = DEFAULT_ZEPTO_PATH
        path = root / path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Zepto products not found: {path}")

    df = pd.read_csv(path, encoding=encoding, on_bad_lines=on_bad_lines)
    cfg = column_config or _load_zepto_column_config()
    zepto_cols = cfg.get("zepto_columns") or {}
    defaults = cfg.get("defaults") or {}

    # Rename Zepto columns to internal names
    rename = {}
    for zepto_name, internal_name in (zepto_cols or {}).items():
        if zepto_name in df.columns and internal_name not in ("sale_price", "market_price", "weight_in_gms"):
            rename[zepto_name] = internal_name
    if rename:
        df = df.rename(columns=rename)

    # Ensure required internal columns exist (in case mapping didn't include them)
    if "product" not in df.columns and "name" in df.columns:
        df["product"] = df["name"].astype(str)
    if CATEGORY_COL not in df.columns and "Category" in df.columns:
        df[CATEGORY_COL] = df["Category"].astype(str)

    # Price: Zepto stores ×100
    if "discountedSellingPrice" in df.columns:
        df["sale_price"] = (df["discountedSellingPrice"].astype(float) / 100.0).round(2)
    if "mrp" in df.columns:
        df["market_price"] = (df["mrp"].astype(float) / 100.0).round(2)

    # Weight
    if "weightInGms" in df.columns:
        w = df["weightInGms"].fillna(0)
        try:
            df["weight_in_gms"] = w.astype(int)
        except (ValueError, TypeError):
            df["weight_in_gms"] = 0
    else:
        df["weight_in_gms"] = 0

    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    # Integer index for retriever/cart
    df = df.reset_index(drop=True)
    df["index"] = range(1, len(df) + 1)

    if "product" not in df.columns or CATEGORY_COL not in df.columns:
        raise ValueError(f"Zepto load failed: need product and {CATEGORY_COL}. Columns: {list(df.columns)}")
    return df


def load_products(
    path: Optional[str] = None,
    source: Optional[str] = None,
    encoding: str = "latin-1",
    on_bad_lines: str = "warn",
) -> pd.DataFrame:
    """
    Load product catalogue. Source 'zepto' or 'bigbasket' (default from config).

    Returns
    -------
    pd.DataFrame
        Unified schema: index, product, category, brand, sale_price, (weight_in_gms for Zepto),
        market_price, type, rating, description, sub_category.
    """
    src = (source or _get_products_source()).strip().lower()
    if src == "zepto":
        return load_zepto_products(path=path, encoding=encoding or "utf-8", on_bad_lines=on_bad_lines)

    # BigBasket
    if path is None:
        root = get_project_root()
        config_path = root / "config" / "paths.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                path = config.get("bigbasket_products", DEFAULT_PRODUCTS_PATH)
            except Exception:
                path = DEFAULT_PRODUCTS_PATH
        else:
            path = DEFAULT_PRODUCTS_PATH
        path = root / path

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Products data not found: {path}")

    df = pd.read_csv(path, encoding=encoding, on_bad_lines=on_bad_lines)

    if "product" not in df.columns or CATEGORY_COL not in df.columns:
        raise ValueError(f"Expected 'product' and '{CATEGORY_COL}'. Columns: {list(df.columns)}")
    return df


def filter_food_products(
    df: pd.DataFrame,
    food_categories: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    Return only rows whose category is in the food categories set.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded products dataframe.
    food_categories : set[str] or None
        If None, loaded from config/food_categories.txt.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe (copy).
    """
    if food_categories is None:
        food_categories = load_food_categories(source=_get_products_source())
    if not food_categories:
        return df.copy()
    mask = df[CATEGORY_COL].astype(str).str.strip().isin(food_categories)
    return df.loc[mask].copy()


if __name__ == "__main__":
    df = load_products()
    print("Products shape:", df.shape)
    print("Category value counts (top 10):")
    print(df[CATEGORY_COL].value_counts().head(10))

    categories = load_food_categories()
    print("\nFood categories from config:", len(categories))
    df_food = filter_food_products(df)
    print("Food products count:", len(df_food))
