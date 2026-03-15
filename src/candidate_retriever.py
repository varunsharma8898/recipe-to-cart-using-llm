"""
Candidate Retriever: BM25 over food products → top-k candidates per ingredient.
Supports config-based retrieval_method: "bm25" (default) or "llm" (BM25 pool + LLM selects top-k).
"""
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from rank_bm25 import BM25Okapi

from .load_products import load_products, filter_food_products, get_project_root, _get_products_source
from .ingredient_synonyms import get_expansion_terms_for_query
from .reranker import llm_select_top_k


def _tokenize(text: str) -> list[str]:
    """Simple tokenize: lowercase, split on non-alphanumeric, min length 2."""
    if not text or not isinstance(text, str):
        return []
    text = str(text).lower().strip()
    tokens = re.findall(r"[a-z0-9]{2,}", text)
    return tokens


def _doc_for_product(row: pd.Series, include_description: bool = True, desc_max_chars: int = 150) -> str:
    """One searchable string per product: name + optional description snippet."""
    parts = [str(row.get("product", "")).strip()]
    if include_description and "description" in row.index and pd.notna(row.get("description")):
        desc = str(row["description"])[:desc_max_chars].strip()
        if desc:
            parts.append(desc)
    return " ".join(parts)


class CandidateRetriever:
    """
    BM25 index over food products. Query by ingredient (or phrase) → top-k product rows.
    When retrieval_method is "llm", uses BM25 to get a pool then LLM to select top-k.
    """

    def __init__(
        self,
        df_products: Optional[pd.DataFrame] = None,
        use_food_filter: bool = True,
        include_description: bool = True,
        desc_max_chars: int = 150,
        *,
        retrieval_method: str = "bm25",
        llm_retrieval_pool_size: int = 30,
    ):
        if df_products is None:
            df_products = load_products(source=_get_products_source())
            if use_food_filter:
                df_products = filter_food_products(df_products)
        self.df = df_products.reset_index(drop=True)
        self._doc_strings = [
            _doc_for_product(self.df.iloc[i], include_description, desc_max_chars)
            for i in range(len(self.df))
        ]
        self._tokenized = [_tokenize(d) for d in self._doc_strings]
        self.bm25 = BM25Okapi(self._tokenized)
        self.retrieval_method = (retrieval_method or "bm25").strip().lower()
        self.llm_retrieval_pool_size = max(1, int(llm_retrieval_pool_size))

    def _retrieve_bm25(
        self,
        query: str,
        k: int,
        expand_query_synonyms: bool = True,
    ) -> list[dict[str, Any]]:
        """Internal: BM25-only retrieval, returns up to k candidate dicts."""
        if not query or not query.strip():
            return []
        search_text = query.strip()
        if expand_query_synonyms:
            expansion = get_expansion_terms_for_query(query)
            if expansion:
                search_text = f"{search_text} {' '.join(expansion)}"
        q_tokens = _tokenize(search_text)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        out = []
        for i in top_indices:
            if scores[i] <= 0:
                break
            row = self.df.iloc[i]
            item = {
                "index": int(row.get("index", i)),
                "product": str(row.get("product", "")),
                "category": str(row.get("category", "")),
                "brand": str(row.get("brand", "")),
                "sale_price": float(row.get("sale_price", 0)),
                "rating": row.get("rating"),
                "type": str(row.get("type", "")),
                "description": (str(row.get("description", ""))[:200] if pd.notna(row.get("description")) else ""),
            }
            if "weight_in_gms" in row.index and pd.notna(row.get("weight_in_gms")):
                try:
                    item["weight_in_gms"] = int(row["weight_in_gms"])
                except (TypeError, ValueError):
                    item["weight_in_gms"] = 0
            out.append(item)
        return out

    def retrieve(
        self,
        query: str,
        k: int = 10,
        expand_query_synonyms: bool = True,
        recipe_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Return top-k products for the query (ingredient or phrase).
        If retrieval_method is "bm25", uses BM25 only. If "llm", gets a BM25 pool then
        LLM selects and orders the top-k.
        If expand_query_synonyms is True, adds canonical and alias terms for BM25.
        Returns list of dicts with keys: index, product, category, brand, sale_price, rating,
        description (truncated), and weight_in_gms when present (Zepto).
        """
        if self.retrieval_method == "llm":
            pool_size = min(self.llm_retrieval_pool_size, max(k, 10))
            pool = self._retrieve_bm25(query, k=pool_size, expand_query_synonyms=expand_query_synonyms)
            if len(pool) <= k:
                return pool
            return llm_select_top_k(query, pool, k=k, recipe_name=recipe_name or "")
        return self._retrieve_bm25(query, k=k, expand_query_synonyms=expand_query_synonyms)


def load_food_products_cached(source: Optional[str] = None) -> pd.DataFrame:
    """Load food products. If source is zepto, load from loader (no BigBasket cache). Else use output/products_food.csv if present."""
    src = (source or _get_products_source()).strip().lower()
    if src == "zepto":
        return filter_food_products(load_products(source="zepto"))
    root = get_project_root()
    cached = root / "output" / "products_food.csv"
    if cached.exists():
        return pd.read_csv(cached, encoding="latin-1", on_bad_lines="warn")
    return filter_food_products(load_products(source="bigbasket"))
