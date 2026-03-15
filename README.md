# Recipe-to-Shopping Cart Generation for Quick Commerce

Recipe name in → cart out. Uses a local LLM (Gemma3 via Ollama) for ingredient extraction and re-ranking, and BM25 for product retrieval. Supports BigBasket and Zepto-style catalogues.

**config/README.md** for all config options.

---

## Datasets

- **RecipeData.csv** – recipes (ingredients, cuisine, etc.).
- **Product catalogue** – pick one in `config/paths.yaml`:
  - **BigBasketProducts.csv** – `dataset/BigBasketProducts.csv`
  - **Zepto** – `dataset/zepto/zepto_v2.csv` (prices in CSV are ×100; has `weightInGms` for quantity)

Set **`products_source: "zepto"`** or **`"bigbasket"`** in `config/paths.yaml`. Zepto setup is described in `docs/PLAN_ZEPTO_MIGRATION.md`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ollama with **gemma3:4b** is needed for ingredient extraction, normalisation, and re-ranking. See `config/ollama.yaml`.

---

## What’s in the repo

**Data and config**

- `config/paths.yaml`, `config/food_categories.txt`; for Zepto, `config/zepto_column_mapping.yaml`
- `src/load_recipes.py`, `src/load_products.py` – load recipes and products (unified for bigbasket/zepto)
- `scripts/prepare_data.py` – build food-product CSV and 50-recipe evaluation set

**Outputs from prepare_data:**  
`output/products_food.csv` (or `products_food_zepto.csv`) and `output/evaluation_set_50_recipes.csv`.

**Ingredient extraction**

- `src/ingredient_extractor.py`, `src/quantity_normaliser.py`, `src/ollama_client.py`
- `src/recipe_to_ingredients.py` – `recipe_name_to_ingredients(name)`; fallback to Cleaned-Ingredients if LLM fails
- `scripts/extract_ingredients.py` – run extraction for one recipe  
  `notebooks/02_ingredient_extraction.ipynb` for checks

**Retrieval and cart**

- `src/candidate_retriever.py` – BM25 over food products; optional query expansion from `config/ingredient_synonyms.yaml`
- `src/product_form_filter.py` – prefer base form over juice/butter/powder etc. (`config/ingredient_form_rules.yaml`)
- `src/reranker.py` – LLM picks best product from BM25 top-k per ingredient
- `src/quantity_optimiser.py` – unit conversion (`config/unit_conversion.yaml`); for Zepto uses `weight_in_gms` when need is in grams; aggregates same product
- `src/build_cart.py` – synonym merge then build_cart(recipe_name) → cart

**Evaluation**

- `src/evaluate.py` – mAP@k on 50-recipe set, failure list (zero BM25 candidates), efficiency (latency per recipe/ingredient), quantity verification sample
- `scripts/run_evaluation.py` – runs the above; `--quick` for fewer recipes

**End-to-end**

- `scripts/run_pipeline.py` – recipe name(s) → cart (JSON/CSV). Multiple recipes → one merged cart (quantities combined per product). Optional `--eval` runs evaluation.

---

## How to run

From project root, Python 3.10+, Ollama running with **gemma3:4b**:

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/run_pipeline.py --recipe "Masala Karela Recipe" --cart-out output/carts/cart.json
```

Optional evaluation (mAP, failures, efficiency, quantity sample):

```bash
python scripts/run_pipeline.py --recipe "Masala Karela Recipe" --eval --quick
```

Other useful commands:

```bash
python scripts/run_pipeline.py --list-recipes 5
python scripts/run_pipeline.py --recipe "Recipe A" --recipe "Recipe B" --cart-out output/carts/cart.json
python scripts/run_pipeline.py --recipe "Name" --cart-out output/carts/cart.csv --format csv
```

Single-recipe cart without full pipeline (if you already have food products and eval set):

```bash
python scripts/recipe_to_cart.py "Masala Karela Recipe"
```

**Fill synonym file from recipe ingredients (Ollama required):**

```bash
python scripts/fill_ingredient_synonyms.py
python scripts/fill_ingredient_synonyms.py --limit 50 --dry-run
```

Config: `config/README.md`. Paths and `products_source` in `config/paths.yaml`; LLM in `config/ollama.yaml`; synonyms in `config/ingredient_synonyms.yaml`.

---

## Project structure

```
config/       paths, food_categories, zepto_column_mapping, ollama, params,
              unit_conversion, ingredient_form_rules, ingredient_synonyms (see config/README.md)
dataset/      RecipeData.csv, BigBasketProducts.csv, zepto/zepto_v2.csv
notebooks/    01 explore data, 02 ingredient extraction, 03 retrieval/cart, 05 evaluation
output/       products_food*.csv, evaluation_set_50_recipes.csv, evaluation/, carts/
scripts/      prepare_data, extract_ingredients, recipe_to_cart, run_evaluation, run_pipeline, fill_ingredient_synonyms
src/          load_recipes, load_products, ingredient_extractor, quantity_normaliser,
              candidate_retriever, product_form_filter, reranker, quantity_optimiser, build_cart, evaluate
```
