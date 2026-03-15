#!/usr/bin/env python3
"""
Fill config/ingredient_synonyms.yaml with Indian/regional names for ingredients.
Gets unique ingredients from the recipe dataset (Cleaned-Ingredients), then uses an LLM
to suggest canonical name and Indian aliases for each. Merges results into the existing
synonym file without overwriting existing entries (unless --overwrite).
Run from project root: python scripts/fill_ingredient_synonyms.py
Requires: Ollama running with model from config/ollama.yaml.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from src.load_recipes import load_recipes, parse_cleaned_ingredients, CLEANED_INGREDIENTS_COL
from src.ollama_client import chat


SYNONYMS_PATH = ROOT / "config" / "ingredient_synonyms.yaml"
BATCH_SIZE = 15  # ingredients per LLM call
SYSTEM_PROMPT = """You are an expert in Indian cooking and grocery names.
Given a list of ingredient names (as they appear in recipes), for each ingredient output:
1. A canonical name (standard English or common name, e.g. "gram flour", "cumin").
2. Indian/regional names (Hindi, regional languages) as aliases, e.g. besan, jeera, dhania.
Output ONLY a valid JSON array. Each element: {"ingredient": "<as given>", "canonical": "<canonical name>", "aliases": ["<alias1>", "<alias2>"]}.
Use lowercase. If an ingredient has no common Indian names, set "aliases" to [] or omit it.
Do not include quantity or unit in canonical/aliases. No explanation."""


def get_unique_ingredients_from_recipes(limit=None):
    """Load recipe dataset and return unique ingredient names from Cleaned-Ingredients."""
    df = load_recipes()
    if CLEANED_INGREDIENTS_COL not in df.columns:
        return []
    all_ingredients = []
    for s in df[CLEANED_INGREDIENTS_COL].dropna():
        parts = parse_cleaned_ingredients(str(s))
        for p in parts:
            # Strip leading numbers and units (e.g. "2 tbsp Salt" -> "Salt")
            cleaned = re.sub(r"^[\d.\s]+", "", p).strip()
            cleaned = re.sub(r"^(tbsp|tsp|cup|cups|gram|g|kg|ml)\s+", "", cleaned, flags=re.I).strip()
            if cleaned and len(cleaned) > 1:
                all_ingredients.append(cleaned)
    unique = list(dict.fromkeys(all_ingredients))  # preserve order, dedupe
    if limit is not None:
        unique = unique[: int(limit)]
    return unique


def load_existing_synonyms() -> dict[str, list[str]]:
    """Load current ingredient_synonyms.yaml as canonical -> list of aliases."""
    if not SYNONYMS_PATH.exists():
        return {}
    with open(SYNONYMS_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out = {}
    for canonical, aliases in data.items():
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        c = canonical.strip()
        if isinstance(aliases, list):
            out[c] = [str(a).strip() for a in aliases if a and str(a).strip()]
        else:
            out[c] = []
    return out


def parse_llm_synonym_response(text: str) -> list[dict]:
    """Parse LLM JSON array response into list of {ingredient, canonical, aliases}."""
    text = text.strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        text = m.group(0)
    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        return []
    out = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        ing = (item.get("ingredient") or item.get("canonical") or "").strip()
        canonical = (item.get("canonical") or ing or "").strip().lower()
        aliases = item.get("aliases")
        if isinstance(aliases, list):
            aliases = [str(a).strip().lower() for a in aliases if a and str(a).strip()]
        else:
            aliases = []
        if canonical:
            out.append({"ingredient": ing, "canonical": canonical, "aliases": aliases})
    return out


def fetch_synonyms_batch(ingredients_batch: list[str]) -> list[dict]:
    """Ask LLM for canonical + Indian aliases for a batch of ingredients."""
    user_content = "Ingredients:\n" + "\n".join(f"- {i}" for i in ingredients_batch)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        response = chat(messages)
        return parse_llm_synonym_response(response)
    except Exception as e:
        print(f"  LLM error: {e}", file=sys.stderr)
        return []


def merge_synonyms(
    existing: dict[str, list[str]],
    new_results: list[dict],
    overwrite: bool = False,
) -> dict[str, list[str]]:
    """Merge new canonical/aliases into existing map. Canonical keys are lowercase."""
    merged = dict(existing) if not overwrite else {}
    for r in new_results:
        canonical = (r.get("canonical") or "").strip().lower()
        aliases = list(r.get("aliases") or [])
        if not canonical:
            continue
        # Dedupe and ensure canonical not in aliases
        aliases = list(dict.fromkeys(a for a in aliases if a and a != canonical))
        if overwrite or canonical not in merged:
            merged[canonical] = aliases
        else:
            existing_aliases = set(merged[canonical])
            for a in aliases:
                existing_aliases.add(a)
            merged[canonical] = list(existing_aliases)
    return merged


def write_synonyms_yaml(data: dict[str, list[str]], path: Path) -> None:
    """Write canonical -> aliases to YAML (sorted by key)."""
    ordered = dict(sorted(data.items(), key=lambda x: x[0].lower()))
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Map ingredient aliases to a canonical name so duplicates merge to one cart line.\n")
        f.write("# Keys: canonical name. Values: list of aliases (Indian/regional names).\n\n")
        yaml.dump(ordered, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fill ingredient_synonyms.yaml with Indian names from recipe dataset + LLM.")
    parser.add_argument("--limit", type=int, default=None, help="Max unique ingredients to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write file; print what would be added.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing synonym entries with LLM output.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Ingredients per LLM call (default {BATCH_SIZE})")
    args = parser.parse_args()

    print("Loading recipes and collecting unique ingredients...")
    unique = get_unique_ingredients_from_recipes(limit=args.limit)
    print(f"  Found {len(unique)} unique ingredients.")

    existing = load_existing_synonyms()
    print(f"  Existing synonyms: {len(existing)} canonicals.")

    # Optionally skip ingredients already covered (any existing canonical or alias matches)
    existing_lower = set()
    for c, alist in existing.items():
        existing_lower.add(c.lower())
        for a in alist:
            existing_lower.add(a.lower())

    to_process = [u for u in unique if u.lower() not in existing_lower]
    if args.limit is not None:
        to_process = to_process[: args.limit]
    print(f"  To process (not already in synonyms): {len(to_process)}.")

    if not to_process:
        print("Nothing new to add. Exiting.")
        return

    all_new = []
    for i in range(0, len(to_process), args.batch_size):
        batch = to_process[i : i + args.batch_size]
        print(f"  Asking LLM for batch {i // args.batch_size + 1} ({len(batch)} ingredients)...")
        results = fetch_synonyms_batch(batch)
        all_new.extend(results)

    merged = merge_synonyms(existing, all_new, overwrite=args.overwrite)
    new_canonicals = len(merged) - len(existing) if not args.overwrite else len(merged)
    print(f"  Merged: {len(merged)} canonicals ({new_canonicals} new or updated).")

    if args.dry_run:
        print("\n[DRY RUN] Would write the following new/updated entries:")
        for r in all_new:
            c = (r.get("canonical") or "").strip().lower()
            if c and (args.overwrite or c not in existing):
                print(f"  {c}: {r.get('aliases', [])}")
        return

    write_synonyms_yaml(merged, SYNONYMS_PATH)
    print(f"Wrote {SYNONYMS_PATH}")


if __name__ == "__main__":
    main()
