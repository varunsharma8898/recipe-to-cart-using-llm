"""
Down-rank products that are a derived form (juice, butter, powder) when the ingredient
implies base/whole form. Used after BM25 retrieve() so the re-ranker sees better ordering.
"""
from pathlib import Path
from typing import Any, Optional

import yaml


def _get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "config").is_dir():
            return current
        current = current.parent
    return current.parent


def load_form_rules(path: Optional[Path] = None) -> dict[str, Any]:
    if path is None:
        path = _get_project_root() / "config" / "ingredient_form_rules.yaml"
    if not Path(path).exists():
        return {"derived_form_keywords": ["juice", "butter", "powder", "paste", "spread"], "base_form_preferred_for": []}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _ingredient_suggests_base_form(ingredient: str, rules: dict[str, Any]) -> bool:
    """True if we should prefer base/whole form (e.g. no juice/butter/powder)."""
    ing_lower = (ingredient or "").strip().lower()
    if not ing_lower:
        return True
    derived = set((rules.get("derived_form_keywords") or []))
    # If ingredient already contains a derived-form word, don't force base
    for d in derived:
        if d in ing_lower:
            return False
    base_preferred = set((rules.get("base_form_preferred_for") or []))
    tokens = set(ing_lower.split())
    if base_preferred and tokens & base_preferred:
        return True
    # Single-word or two-word ingredient without derived keyword → prefer base
    if len(tokens) <= 2:
        return True
    return False


def _product_is_derived_form(product_name: str, rules: dict[str, Any]) -> bool:
    """True if product name indicates a derived form (juice, butter, powder, etc.)."""
    name_lower = (product_name or "").lower()
    derived = rules.get("derived_form_keywords") or []
    for kw in derived:
        if kw in name_lower:
            return True
    return False


def _product_is_wrong_type_for_ingredient(ingredient: str, product_name: str, rules: dict[str, Any]) -> bool:
    """True if product is a wrong type for this ingredient (e.g. cashew cookies for cashew, upma for oil, rice bran oil for rice)."""
    ing_lower = (ingredient or "").strip().lower()
    name_lower = (product_name or "").lower()
    if not ing_lower:
        return False
    avoid_map = rules.get("ingredient_avoid_product_keywords") or {}
    for _key, spec in avoid_map.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("exclude"):
            continue  # Handled by _product_should_exclude; don't count as wrong_type
        tokens = spec.get("ingredient_tokens") or []
        avoid = spec.get("product_avoid") or []
        if not tokens or not avoid:
            continue
        if not any(t in ing_lower for t in tokens):
            continue
        if any(a in name_lower for a in avoid):
            return True
    return False


def _product_should_exclude_for_ingredient(ingredient: str, product_name: str, rules: dict[str, Any]) -> bool:
    """True if this product must be excluded entirely for this ingredient (e.g. besan for maida)."""
    ing_lower = (ingredient or "").strip().lower()
    name_lower = (product_name or "").lower()
    if not ing_lower:
        return False
    avoid_map = rules.get("ingredient_avoid_product_keywords") or {}
    for _key, spec in avoid_map.items():
        if not isinstance(spec, dict) or not spec.get("exclude"):
            continue
        tokens = spec.get("ingredient_tokens") or []
        avoid = spec.get("product_avoid") or []
        if not tokens or not avoid:
            continue
        if not any(t in ing_lower for t in tokens):
            continue
        if any(a in name_lower for a in avoid):
            return True
    return False


def reject_product_for_ingredient(ingredient: str, product_name: str, rules: Optional[dict[str, Any]] = None) -> bool:
    """True if this product must not be chosen for this ingredient (e.g. besan for maida)."""
    return _product_should_exclude_for_ingredient(
        ingredient, product_name or "", rules or load_form_rules()
    )


def reorder_candidates(
    candidates: list[dict[str, Any]],
    ingredient: str,
    rules: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Reorder candidates so that derived-form products (juice, butter, powder, etc.)
    are placed after base-form products when the ingredient suggests base form.
    Preserves relative order within each group.
    """
    if not candidates:
        return candidates
    rules = rules or load_form_rules()
    base, derived, wrong_type = [], [], []
    for c in candidates:
        name = c.get("product", "")
        if _product_should_exclude_for_ingredient(ingredient, name, rules):
            continue  # Drop entirely (e.g. besan when ingredient is maida)
        if _product_is_wrong_type_for_ingredient(ingredient, name, rules):
            wrong_type.append(c)
        elif _ingredient_suggests_base_form(ingredient, rules) and _product_is_derived_form(name, rules):
            derived.append(c)
        else:
            base.append(c)
    return base + derived + wrong_type
