# Recipe-to-Shopping Cart Generation for Quick Commerce

Recipe name in → cart out. Uses a **local LLM** (Gemma3 via **Ollama**) for ingredient extraction, normalisation, and re-ranking, and **BM25** for product retrieval. Supports **BigBasket** and **Zepto**-style catalogues.

Planning and config reference: **PROJECT_PLAN.md**, **NEXT_STEPS.md**, and **config/README.md**.

---

## Datasets

- **RecipeData.csv** – recipes (ingredients, cuisine, etc.).
- **Product catalogue** – choose in `config/paths.yaml`:
  - **BigBasket** – `dataset/BigBasketProducts.csv`
  - **Zepto** – `dataset/zepto/zepto_v2.csv` (prices in CSV are ×100; includes `weightInGms` for quantity logic)

Set `products_source: "zepto"` or `"bigbasket"` in `config/paths.yaml`. Zepto migration notes: `docs/PLAN_ZEPTO_MIGRATION.md`.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Ollama** with **gemma3:4b** (default) is used for extraction, normalisation, and re-ranking. Pull models as needed:

```bash
ollama pull gemma3:4b
```

Runtime URL and model name: `config/ollama.yaml`.

---

## How to run

From the project root, **Python 3.10+**, with Ollama running:

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/run_pipeline.py --recipe "Masala Karela Recipe" --cart-out output/carts/cart.json
```

**Evaluation** (mAP, failures, efficiency, quantity sample):

```bash
python scripts/run_pipeline.py --recipe "Masala Karela Recipe" --eval --quick
```

Or:

```bash
PYTHONPATH=. python scripts/run_evaluation.py --quick
```

**Multi-model comparison** (Gemma3 1B / 4B / 12B):

```bash
ollama pull gemma3:1b && ollama pull gemma3:4b && ollama pull gemma3:12b
for m in gemma3:1b gemma3:4b gemma3:12b; do PYTHONPATH=. python scripts/run_evaluation.py --model "$m" --quick; done
PYTHONPATH=. python scripts/compare_model_evaluations.py --quick
```

Results under `output/evaluation/` (per-model subfolders when using `--model`). Details: **docs/EVALUATION_MODEL_COMPARISON.md**.

**More examples:**

```bash
python scripts/run_pipeline.py --list-recipes 5
python scripts/run_pipeline.py --recipe "Recipe A" --recipe "Recipe B" --cart-out output/carts/cart.json
python scripts/run_pipeline.py --recipe "Name" --cart-out output/carts/cart.csv --format csv
python scripts/recipe_to_cart.py "Masala Karela Recipe"
```

**Synonym helper** (Ollama required):

```bash
python scripts/fill_ingredient_synonyms.py
python scripts/fill_ingredient_synonyms.py --limit 50 --dry-run
```

---

## Tests

Automated tests (pytest) cover recipe/product loading, LLM output parsing, retrieval, filtering/ranking, quantity logic, and end-to-end cart building:

```bash
pytest tests/ -q
```

---

## Repository layout

```
config/           paths, ollama, params, YAML rules (synonyms, units, form rules) — see config/README.md
dataset/          RecipeData.csv, BigBasketProducts.csv, zepto/zepto_v2.csv
docs/             analysis, methodology, evaluation, reports (see table above)
notebooks/        data exploration, extraction, retrieval/cart, evaluation
output/           products_food*.csv, evaluation_set_50_recipes.csv, evaluation/, carts/
scripts/          prepare_data, extract_ingredients, recipe_to_cart, run_pipeline,
                  run_evaluation, compare_model_evaluations, fill_ingredient_synonyms
src/              loaders, ingredient_extractor, quantity_normaliser, ollama_client,
                  candidate_retriever, product_form_filter, reranker, quantity_optimiser,
                  build_cart, evaluate
tests/            pytest suite
```

---

## Pipeline overview

| Stage | Role |
| ----- | ---- |
| **LLM** | Ingredient extraction (JSON), quantity normalisation, optional LLM-assisted top-*k* from BM25 pool, re-ranking |
| **BM25** | Lexical retrieval over product text with optional synonym expansion |
| **Rules** | Product-form filter (`ingredient_form_rules.yaml`), unit conversion (`unit_conversion.yaml`) |
| **Output** | Cart as JSON or CSV with product, quantity, price, ingredient mapping; multi-recipe aggregation |

Tunable behaviour: `config/params.yaml` (retrieval mode, reranker, top-*k*, etc.).
