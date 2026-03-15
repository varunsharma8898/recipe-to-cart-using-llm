"""
Merge ingredients that refer to the same product (e.g. besan and gram flour) so the cart
has one line per product instead of duplicates.
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


def _load_synonym_yaml(path: Optional[Path] = None) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Load YAML; return (alias_lower -> canonical, canonical_lower -> [canonical, alias1, ...])."""
    if path is None:
        path = _get_project_root() / "config" / "ingredient_synonyms.yaml"
    alias_to_canonical: dict[str, str] = {}
    canonical_to_terms: dict[str, list[str]] = {}
    if not Path(path).exists():
        return alias_to_canonical, canonical_to_terms
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    for canonical, aliases in data.items():
        if not aliases:
            continue
        c = (canonical or "").strip()
        if not c:
            continue
        alias_to_canonical[c.lower()] = c
        terms = [c] + [a.strip() for a in aliases if a and isinstance(a, str)]
        canonical_to_terms[c.lower()] = terms
        for a in aliases:
            if a and isinstance(a, str):
                alias_to_canonical[a.strip().lower()] = c
    return alias_to_canonical, canonical_to_terms


def load_synonym_map(path: Optional[Path] = None) -> dict[str, str]:
    """
    Load config and return alias (lower) -> canonical name.
    Canonical name is kept as in file (e.g. 'gram flour'); lookup is by lowercase.
    """
    out, _ = _load_synonym_yaml(path)
    return out


def get_expansion_terms_for_query(query: str, path: Optional[Path] = None) -> list[str]:
    """
    Return synonym/alias terms to add to a BM25 query so products named with
    alternatives (e.g. 'Gram flour') match when the user query is 'besan'.
    Returns the canonical name and all aliases for the query's canonical; empty if no synonym.
    """
    if not query or not isinstance(query, str):
        return []
    alias_to_canonical, canonical_to_terms = _load_synonym_yaml(path)
    key = query.strip().lower()
    canonical = alias_to_canonical.get(key)
    if canonical is None:
        return []
    return canonical_to_terms.get(canonical.lower(), [canonical])


# Single canonical for cooking oils so "sunflower oil", "vegetable oil", "oil" merge to one cart line.
_OIL_CANONICAL = "vegetable oil"
_OIL_ALIASES = {"sunflower oil", "vegetable oil", "cooking oil", "refined oil", "oil", "tel", "refined sunflower oil"}


def canonicalise_ingredient_name(name: str, synonym_map: Optional[dict[str, str]] = None) -> str:
    """Return canonical ingredient name for display/grouping; if no synonym, return name as-is."""
    if not name or not isinstance(name, str):
        return name or ""
    synonym_map = synonym_map or load_synonym_map()
    key = name.strip().lower()
    canonical = synonym_map.get(key, name.strip())
    # Dedupe oils: all cooking oil variants map to one canonical so they merge across recipes
    if key in _OIL_ALIASES or (canonical and canonical.strip().lower() in _OIL_ALIASES):
        return _OIL_CANONICAL
    return canonical


def merge_ingredients_by_synonym(
    ingredients: list[dict[str, Any]],
    synonym_map: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """
    Group ingredients by canonical name (using synonym map) and merge quantities.
    Same unit: quantities are summed. Different units: first unit is kept, quantities summed as number.
    """
    if not ingredients:
        return []
    synonym_map = synonym_map or load_synonym_map()
    groups: dict[str, list[dict[str, Any]]] = {}
    for ing in ingredients:
        name = ing.get("ingredient", "") or ""
        canonical = canonicalise_ingredient_name(name, synonym_map)
        if canonical not in groups:
            groups[canonical] = []
        groups[canonical].append(ing)
    merged = []
    for canonical, group in groups.items():
        if len(group) == 1:
            merged.append({**group[0], "ingredient": canonical})
            continue
        q_sum = 0
        unit_used = group[0].get("unit", "piece")
        form_used = group[0].get("form", "base")
        for g in group:
            q = g.get("quantity", 1)
            u = g.get("unit", "piece")
            try:
                q_val = float(q)
            except (TypeError, ValueError):
                q_val = 1
            if u == unit_used:
                q_sum += q_val
            else:
                q_sum += q_val
        merged.append({
            "ingredient": canonical,
            "quantity": q_sum,
            "unit": unit_used,
            "form": form_used,
        })
    return merged
