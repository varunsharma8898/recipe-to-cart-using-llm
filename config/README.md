# Config

- **paths.yaml** – Dataset and output paths (recipe_data, bigbasket_products, zepto_products, output_dir, evaluation_set, products_food). Set **products_source** to `"zepto"` or `"bigbasket"` to choose the product catalogue (Zepto: prices ×100 in CSV are divided by 100; weightInGms used for quantity).
- **zepto_column_mapping.yaml** – Zepto CSV column names → internal schema (product, category, sale_price, weight_in_gms, etc.).
- **food_categories.txt** – BigBasket category names to include for ingredient matching (one per line).
- **food_categories_zepto.txt** – Zepto category names (used when products_source is zepto; Zepto uses different names than BigBasket).
- **ollama.yaml** – LLM: base_url (default http://localhost:11434), model (e.g. gemma3:4b), timeout.
- **params.yaml** – Pipeline: build_cart (top_k_retrieve, max_candidates_rerank, use_reranker, retrieval_method, llm_retrieval_pool_size). Set **retrieval_method** to `"bm25"` (default) or `"llm"` to use LLM for candidate retrieval (BM25 pool + LLM selects top-k). When `"llm"`, **llm_retrieval_pool_size** is the BM25 pool size before LLM picks top_k_retrieve. With **use_reranker: true** the final product per ingredient is always chosen by the LLM reranker, so the cart can look the same for bm25 vs llm; to see retrieval-only behaviour set **use_reranker: false** (or use `--no-reranker`).
- **unit_conversion.yaml** – Cooking units → metric (volume_ml, mass_g, to_taste). volume_to_mass_dry_goods: convert recipe cups to grams for rice/flour so quantity uses product weight_in_gms.
- **ingredient_form_rules.yaml** – Prefer base/whole product form: derived_form_keywords (juice, butter, powder, etc.), base_form_preferred_for (karela, cashew, moringa, …). Used by product_form_filter and reranker.

All paths in paths.yaml are relative to project root unless absolute.
