"""
Quantity Optimiser: unit conversion to metric, aggregate same product across recipes,
output cart rows with quantity and price. When product has weight_in_gms (Zepto), quantity = ceil(need_g / weight_in_gms).
Otherwise quantity = units to buy (1 by default) or merge duplicates.
"""
import math
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml


def get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "dataset").is_dir():
            return current
        current = current.parent
    return current.parent


def load_unit_conversion() -> dict[str, Any]:
    """Load config/unit_conversion.yaml."""
    root = get_project_root()
    path = root / "config" / "unit_conversion.yaml"
    if not path.exists():
        return {"volume_ml": {"cup": 240, "tablespoon": 15, "teaspoon": 5}, "mass_g": {"pound": 454, "gram": 1}}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def recipe_quantity_to_metric(quantity: float, unit: str) -> tuple[float, str]:
    """
    Convert recipe quantity to metric. Returns (value, "ml" | "g").
    For "to taste" / "pinch" etc. returns (0, "unit").
    """
    cfg = load_unit_conversion()
    unit_lower = (unit or "").strip().lower()
    to_taste = cfg.get("to_taste") or ["to taste", "as required", "pinch", "optional"]
    if any(t in unit_lower for t in to_taste):
        return 0.0, "unit"
    vol = (cfg.get("volume_ml") or {}).get(unit_lower) or (cfg.get("volume_ml") or {}).get(unit_lower.replace(" ", ""))
    if vol is not None:
        return quantity * float(vol), "ml"
    mass = (cfg.get("mass_g") or {}).get(unit_lower) or (cfg.get("mass_g") or {}).get(unit_lower.replace(" ", ""))
    if mass is not None:
        return quantity * float(mass), "g"
    return quantity, "unit"


def _piece_to_grams_vegetables(ingredient_name: str, pieces: float) -> Optional[float]:
    """
    When recipe gives pieces (e.g. 6 potatoes), convert to grams using piece_to_grams_vegetables
    so quantity uses product weight_in_gms (e.g. 6 potatoes ≈ 600g → 1 unit of 1kg pack).
    """
    if not ingredient_name or pieces <= 0:
        return None
    cfg = load_unit_conversion()
    rules = cfg.get("piece_to_grams_vegetables") or []
    ing_lower = (ingredient_name or "").strip().lower()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        keywords = rule.get("keywords") or []
        g_per_piece = rule.get("grams_per_piece")
        if not keywords or g_per_piece is None:
            continue
        if any(kw in ing_lower for kw in keywords):
            return pieces * float(g_per_piece)
    return None


def _volume_ml_to_grams_for_dry_goods(ingredient_name: str, volume_ml: float) -> Optional[float]:
    """
    When recipe gives volume (cups/tbsp) for a dry good (rice, flour), convert to grams
    so we can use product weight_in_gms to compute quantity. Returns need_grams or None.
    """
    if not ingredient_name or volume_ml <= 0:
        return None
    cfg = load_unit_conversion()
    rules = cfg.get("volume_to_mass_dry_goods") or []
    ing_lower = (ingredient_name or "").strip().lower()
    # 1 cup = 240 ml
    cups = volume_ml / 240.0
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        keywords = rule.get("keywords") or []
        g_per_cup = rule.get("grams_per_cup")
        if not keywords or g_per_cup is None:
            continue
        if any(kw in ing_lower for kw in keywords):
            return cups * float(g_per_cup)
    return None


def _volume_ml_to_grams_powders_condiments(ingredient_name: str, volume_ml: float) -> Optional[float]:
    """
    Recipe in tsp/tbsp (as volume_ml) for powders or condiments (salt, sugar) → grams.
    Uses grams_per_teaspoon (5 ml) and/or grams_per_tablespoon (15 ml).
    """
    if not ingredient_name or volume_ml <= 0:
        return None
    cfg = load_unit_conversion()
    for section in ("volume_to_mass_powders", "volume_to_mass_condiments"):
        rules = cfg.get(section) or []
        ing_lower = (ingredient_name or "").strip().lower()
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            keywords = rule.get("keywords") or []
            g_tsp = rule.get("grams_per_teaspoon")
            g_tbsp = rule.get("grams_per_tablespoon")
            if not keywords or (g_tsp is None and g_tbsp is None):
                continue
            if not any(kw in ing_lower for kw in keywords):
                continue
            # Prefer tbsp (15 ml) then tsp (5 ml) for conversion
            if g_tbsp is not None and volume_ml >= 15:
                return (volume_ml / 15.0) * float(g_tbsp)
            if g_tsp is not None:
                return (volume_ml / 5.0) * float(g_tsp)
            return None
    return None


def _volume_ml_to_grams_nuts(ingredient_name: str, volume_ml: float) -> Optional[float]:
    """Recipe in cups (volume_ml) for nuts → grams. 1 cup = 240 ml."""
    if not ingredient_name or volume_ml <= 0:
        return None
    cfg = load_unit_conversion()
    rules = cfg.get("volume_to_mass_nuts") or []
    ing_lower = (ingredient_name or "").strip().lower()
    cups = volume_ml / 240.0
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        keywords = rule.get("keywords") or []
        g_per_cup = rule.get("grams_per_cup")
        if not keywords or g_per_cup is None:
            continue
        if any(kw in ing_lower for kw in keywords):
            return cups * float(g_per_cup)
    return None


def _volume_ml_to_grams_seeds(ingredient_name: str, volume_ml: float) -> Optional[float]:
    """Whole seeds (cumin seeds, etc.): tsp/tbsp → grams. Tried before powders so 'cumin seeds' matches."""
    if not ingredient_name or volume_ml <= 0:
        return None
    cfg = load_unit_conversion()
    rules = cfg.get("volume_to_mass_seeds") or []
    ing_lower = (ingredient_name or "").strip().lower()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        keywords = rule.get("keywords") or []
        g_tsp = rule.get("grams_per_teaspoon")
        g_tbsp = rule.get("grams_per_tablespoon")
        if not keywords or (g_tsp is None and g_tbsp is None):
            continue
        if not any(kw in ing_lower for kw in keywords):
            continue
        if g_tbsp is not None and volume_ml >= 15:
            return (volume_ml / 15.0) * float(g_tbsp)
        if g_tsp is not None:
            return (volume_ml / 5.0) * float(g_tsp)
        return None
    return None


def _volume_ml_to_need_grams(ingredient_name: str, volume_ml: float) -> Optional[float]:
    """
    Try all volume→grams rules: dry goods (flour, rice), seeds (before powders), powders, condiments, nuts.
    Returns need_grams or None.
    """
    need_g = _volume_ml_to_grams_for_dry_goods(ingredient_name, volume_ml)
    if need_g is not None:
        return need_g
    need_g = _volume_ml_to_grams_seeds(ingredient_name, volume_ml)
    if need_g is not None:
        return need_g
    need_g = _volume_ml_to_grams_powders_condiments(ingredient_name, volume_ml)
    if need_g is not None:
        return need_g
    return _volume_ml_to_grams_nuts(ingredient_name, volume_ml)


def _liquid_ml_per_unit(ingredient_name: str) -> Optional[float]:
    """
    For oils/ghee etc.: recipe in ml, product sold by bottle. Returns ml per unit (e.g. 500) or None.
    """
    if not ingredient_name:
        return None
    cfg = load_unit_conversion()
    rules = cfg.get("liquid_ml_per_unit") or []
    ing_lower = (ingredient_name or "").strip().lower()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        keywords = rule.get("keywords") or []
        ml_per = rule.get("ml_per_unit")
        if not keywords or ml_per is None:
            continue
        if any(kw in ing_lower for kw in keywords):
            return float(ml_per)
    return None


def _default_pack_grams(ingredient_name: str) -> int:
    """
    When product has no weight_in_gms, assume pack size (grams) for quantity = ceil(need_grams / pack).
    Used so we still get sensible quantity and can aggregate by need_grams.
    """
    cfg = load_unit_conversion()
    rules = cfg.get("default_pack_grams") or []
    ing_lower = (ingredient_name or "").strip().lower()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        keywords = rule.get("keywords") or []
        g = rule.get("grams")
        if g is None:
            continue
        if not keywords:
            return max(1, int(g))
        if any(kw in ing_lower for kw in keywords):
            return max(1, int(g))
    return 200


def aggregate_cart_rows(cart_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge rows with the same product index. When both rows have need_grams and pack_grams,
    sum need_grams and set quantity = ceil(total_need_grams / pack_grams) so we don't over-count packs.
    Otherwise sum quantities. Cap at 1 for to-taste/pinch.
    """
    by_index: dict[int, dict[str, Any]] = {}
    for row in cart_rows:
        idx = row.get("index") or row.get("product_index")
        if idx is None:
            continue
        idx = int(idx)
        ing = row.get("ingredient", "") or ""
        if isinstance(row.get("ingredients"), list):
            ing_list = list(row["ingredients"])
        else:
            ing_list = [ing] if ing else []
        need_g = row.get("need_grams")
        pack_g = row.get("pack_grams")
        need_m = row.get("need_ml")
        ml_per = row.get("ml_per_unit")

        if idx not in by_index:
            by_index[idx] = {
                "index": idx,
                "product": row.get("product", ""),
                "brand": row.get("brand", ""),
                "sale_price": float(row.get("sale_price", 0)),
                "quantity": 1 if row.get("cap_quantity_at_1") else row.get("quantity", 1),
                "unit_metric": row.get("unit_metric", "unit"),
                "cap_quantity_at_1": bool(row.get("cap_quantity_at_1")),
                "ingredients": ing_list,
            }
            if need_g is not None and pack_g is not None and pack_g > 0:
                by_index[idx]["_agg_need_grams"] = need_g
                by_index[idx]["_agg_pack_grams"] = pack_g
            if need_m is not None and ml_per is not None and ml_per > 0:
                by_index[idx]["_agg_need_ml"] = need_m
                by_index[idx]["_agg_ml_per_unit"] = ml_per
        else:
            cap1 = by_index[idx].get("cap_quantity_at_1") or row.get("cap_quantity_at_1")
            if cap1:
                by_index[idx]["quantity"] = 1
                by_index[idx]["cap_quantity_at_1"] = True
                for key in ("_agg_need_grams", "_agg_pack_grams", "_agg_need_ml", "_agg_ml_per_unit"):
                    by_index[idx].pop(key, None)
            elif need_m is not None and ml_per is not None and ml_per > 0 and "_agg_need_ml" in by_index[idx]:
                by_index[idx]["_agg_need_ml"] = by_index[idx]["_agg_need_ml"] + need_m
                total_ml = by_index[idx]["_agg_need_ml"]
                mpu = by_index[idx]["_agg_ml_per_unit"]
                by_index[idx]["quantity"] = max(1, math.ceil(total_ml / mpu))
            elif need_g is not None and pack_g is not None and pack_g > 0 and "_agg_need_grams" in by_index[idx]:
                by_index[idx]["_agg_need_grams"] = by_index[idx]["_agg_need_grams"] + need_g
                total_need = by_index[idx]["_agg_need_grams"]
                p = by_index[idx]["_agg_pack_grams"]
                by_index[idx]["quantity"] = max(1, math.ceil(total_need / p))
            else:
                q = by_index[idx]["quantity"]
                q2 = row.get("quantity", 1)
                if isinstance(q, (int, float)) and isinstance(q2, (int, float)):
                    by_index[idx]["quantity"] = q + q2
                else:
                    by_index[idx]["quantity"] = max(1, (q if isinstance(q, (int, float)) else 1) + (q2 if isinstance(q2, (int, float)) else 1))
                for key in ("_agg_need_grams", "_agg_pack_grams", "_agg_need_ml", "_agg_ml_per_unit"):
                    by_index[idx].pop(key, None)
            by_index[idx]["ingredients"] = by_index[idx].get("ingredients", []) + ing_list

    out = []
    for r in by_index.values():
        if "_agg_need_grams" in r and "_agg_pack_grams" in r:
            r["need_grams"] = r["_agg_need_grams"]
            r["pack_grams"] = r["_agg_pack_grams"]
        if "_agg_need_ml" in r and "_agg_ml_per_unit" in r:
            r["need_ml"] = r["_agg_need_ml"]
            r["ml_per_unit"] = r["_agg_ml_per_unit"]
        r = {k: v for k, v in r.items() if k not in ("_agg_need_grams", "_agg_pack_grams", "_agg_need_ml", "_agg_ml_per_unit")}
        out.append(r)
    return out


def optimise_cart(
    selected_products: list[dict[str, Any]],
    ingredients: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build cart list from selected products (one per ingredient) and ingredient quantities.
    Converts to metric where possible; then aggregates duplicate products.

    Parameters
    ----------
    selected_products : list[dict]
        One dict per ingredient (from reranker), with index, product, brand, sale_price, etc.
    ingredients : list[dict]
        From earlier phase: [{"ingredient", "quantity", "unit"}, ...]. Must align 1:1 with selected_products
        (same length; ingredient i → selected_products[i]).

    Returns
    -------
    list[dict]
        Cart: [{"index", "product", "brand", "sale_price", "quantity", "unit_metric"}, ...],
        aggregated so same product appears once with combined quantity.
    """
    if len(selected_products) != len(ingredients):
        # Pad or truncate to min length
        n = min(len(selected_products), len(ingredients))
        selected_products = selected_products[:n]
        ingredients = ingredients[:n]
    cart_rows = []
    for sel, ing in zip(selected_products, ingredients):
        q = ing.get("quantity", 1)
        u = ing.get("unit", "piece")
        ingredient_name = ing.get("ingredient", "") or ""
        try:
            q_val = float(q)
        except (TypeError, ValueError):
            q_val = 1
        value_metric, _recipe_unit = recipe_quantity_to_metric(q_val, u)
        weight_g = sel.get("weight_in_gms") or 0
        try:
            weight_g = int(weight_g) if weight_g else 0
        except (TypeError, ValueError):
            weight_g = 0
        # "To taste" / pinch (salt, etc.): always 1 unit, cap when aggregating
        to_taste = _recipe_unit == "unit" and value_metric == 0
        cap_quantity_at_1 = to_taste
        need_grams: Optional[float] = None
        pack_grams: Optional[float] = None
        need_ml: Optional[float] = None
        ml_per_unit: Optional[float] = None

        if to_taste:
            quantity = 1
        # Use product weight_in_gms to compute quantity when available
        elif weight_g > 0 and value_metric > 0:
            if _recipe_unit == "g":
                need_grams = value_metric
                pack_grams = float(weight_g)
                quantity = max(1, math.ceil(need_grams / pack_grams))
            elif _recipe_unit == "ml":
                need_g = _volume_ml_to_need_grams(ingredient_name, value_metric)
                if need_g is not None:
                    need_grams = need_g
                    pack_grams = float(weight_g)
                    quantity = max(1, math.ceil(need_grams / pack_grams))
                else:
                    quantity = max(1, round(value_metric / 100))
            elif _recipe_unit == "unit":
                need_g = _piece_to_grams_vegetables(ingredient_name, q_val)
                if need_g is not None:
                    need_grams = need_g
                    pack_grams = float(weight_g)
                    quantity = max(1, math.ceil(need_grams / pack_grams))
                else:
                    quantity = max(1, int(q_val)) if q_val >= 1 else 1
            else:
                quantity = max(1, int(q_val)) if q_val >= 1 else 1
        elif _recipe_unit == "ml" and value_metric > 0:
            _ml_per = _liquid_ml_per_unit(ingredient_name)
            if _ml_per is not None and _ml_per > 0:
                need_ml = value_metric
                ml_per_unit = _ml_per
                quantity = max(1, math.ceil(need_ml / ml_per_unit))
            else:
                need_g = _volume_ml_to_need_grams(ingredient_name, value_metric)
                if need_g is not None:
                    need_grams = need_g
                    pack_grams = float(_default_pack_grams(ingredient_name))
                    quantity = max(1, math.ceil(need_grams / pack_grams))
                else:
                    quantity = max(1, round(value_metric / 100))
        elif _recipe_unit == "g" and value_metric > 0:
            need_grams = value_metric
            pack_grams = float(weight_g) if weight_g > 0 else float(_default_pack_grams(ingredient_name))
            quantity = max(1, math.ceil(need_grams / pack_grams))
        elif _recipe_unit == "unit" and q_val >= 1:
            need_g = _piece_to_grams_vegetables(ingredient_name, q_val)
            if need_g is not None:
                need_grams = need_g
                pack_grams = float(weight_g) if weight_g > 0 else float(_default_pack_grams(ingredient_name))
                quantity = max(1, math.ceil(need_grams / pack_grams))
            else:
                quantity = max(1, int(q_val))
        else:
            quantity = max(1, int(q_val)) if q_val >= 1 else 1

        row = {
            "index": sel.get("index"),
            "product": sel.get("product", ""),
            "brand": sel.get("brand", ""),
            "sale_price": sel.get("sale_price", 0),
            "quantity": quantity,
            "unit_metric": "unit",
            "cap_quantity_at_1": cap_quantity_at_1,
            "ingredient": ingredient_name,
        }
        if need_grams is not None and pack_grams is not None and pack_grams > 0:
            row["need_grams"] = need_grams
            row["pack_grams"] = pack_grams
        if need_ml is not None and ml_per_unit is not None and ml_per_unit > 0:
            row["need_ml"] = need_ml
            row["ml_per_unit"] = ml_per_unit
        cart_rows.append(row)
    return aggregate_cart_rows(cart_rows)
