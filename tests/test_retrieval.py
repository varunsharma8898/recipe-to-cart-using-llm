"""
Functional tests: Retrieval (Section 4.6.2).
Verify that BM25 returns non-empty, correctly structured candidate lists when relevant products exist.
"""
import pytest

from src.candidate_retriever import CandidateRetriever


class TestRetrieval:
    """BM25 retrieval tests using minimal product DataFrame."""

    def test_retriever_returns_non_empty_for_relevant_query(self, df_products_mini):
        """BM25 returns non-empty list when products match the query."""
        retriever = CandidateRetriever(
            df_products=df_products_mini,
            use_food_filter=False,
            retrieval_method="bm25",
        )
        candidates = retriever.retrieve("potato", k=5)
        assert len(candidates) > 0
        assert len(candidates) <= 5

    def test_retriever_candidate_structure(self, df_products_mini):
        """Each candidate has index, product, category, brand, sale_price."""
        retriever = CandidateRetriever(
            df_products=df_products_mini,
            use_food_filter=False,
            retrieval_method="bm25",
        )
        candidates = retriever.retrieve("salt", k=3)
        assert len(candidates) >= 1
        for c in candidates:
            assert "index" in c
            assert "product" in c
            assert "category" in c
            assert "sale_price" in c
            assert isinstance(c["sale_price"], (int, float))

    def test_retriever_respects_k(self, df_products_mini):
        """At most k candidates returned."""
        retriever = CandidateRetriever(
            df_products=df_products_mini,
            use_food_filter=False,
            retrieval_method="bm25",
        )
        candidates = retriever.retrieve("vegetable", k=2)
        assert len(candidates) <= 2

    def test_retriever_empty_query_returns_empty(self, df_products_mini):
        """Empty or whitespace query returns empty list."""
        retriever = CandidateRetriever(
            df_products=df_products_mini,
            use_food_filter=False,
            retrieval_method="bm25",
        )
        assert retriever.retrieve("", k=5) == []
        assert retriever.retrieve("   ", k=5) == []

    def test_retriever_relevant_product_ranked_high(self, df_products_mini):
        """Query 'Potato' should return potato product in results."""
        retriever = CandidateRetriever(
            df_products=df_products_mini,
            use_food_filter=False,
            retrieval_method="bm25",
        )
        candidates = retriever.retrieve("Potato", k=5)
        assert len(candidates) >= 1
        products = [c["product"].lower() for c in candidates]
        assert any("potato" in p for p in products)

    def test_retriever_weight_in_gms_present_when_in_df(self, df_products_mini):
        """Candidates include weight_in_gms when present in product data."""
        retriever = CandidateRetriever(
            df_products=df_products_mini,
            use_food_filter=False,
            retrieval_method="bm25",
        )
        candidates = retriever.retrieve("potato", k=1)
        assert len(candidates) >= 1
        assert "weight_in_gms" in candidates[0]
        assert candidates[0]["weight_in_gms"] == 1000
