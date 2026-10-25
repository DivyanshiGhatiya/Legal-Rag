import unittest
from unittest.mock import patch

from utils import embedding, retrieval


class ResourceCacheTests(unittest.TestCase):
    def tearDown(self):
        embedding._get_ollama_client.cache_clear()
        embedding._get_collection.cache_clear()
        embedding._get_chroma_client.cache_clear()

    def test_ollama_client_is_reused(self):
        embedding._get_ollama_client.cache_clear()
        with patch.object(embedding.ollama, "Client") as client_factory:
            first = embedding._get_ollama_client()
            second = embedding._get_ollama_client()

        self.assertIs(first, second)
        client_factory.assert_called_once_with(host="http://127.0.0.1:11434")

    def test_chroma_client_and_collection_are_reused(self):
        embedding._get_collection.cache_clear()
        embedding._get_chroma_client.cache_clear()
        with patch.object(embedding.chromadb, "PersistentClient") as client_factory:
            first = embedding.get_collection("cache_test", "cache_test_dir")
            second = embedding.get_collection("cache_test", "cache_test_dir")

        self.assertIs(first, second)
        client_factory.assert_called_once()
        client_factory.return_value.get_or_create_collection.assert_called_once_with(
            name="cache_test"
        )

    def test_bm25_corpus_is_loaded_once_per_index_cache_key(self):
        retrieval._bm25_cache.pop(("cache_test", "cache_test_dir"), None)
        collection = type(
            "Collection",
            (),
            {
                "get": lambda self, **kwargs: {
                    "ids": ["chunk-1"],
                    "documents": ["legal evidence"],
                    "metadatas": [{"source_name": "contract.pdf"}],
                }
            },
        )()
        with patch.object(retrieval, "get_collection", return_value=collection) as get:
            first = retrieval._load_corpus("cache_test", "cache_test_dir")
            second = retrieval._load_corpus("cache_test", "cache_test_dir")

        self.assertIs(first, second)
        get.assert_called_once()

    def test_cross_encoder_is_loaded_once(self):
        existing_encoder = retrieval._cross_encoder
        retrieval._cross_encoder = None
        try:
            with patch.object(retrieval, "CrossEncoder") as encoder_factory:
                first = retrieval._get_cross_encoder()
                second = retrieval._get_cross_encoder()

            self.assertIs(first, second)
            encoder_factory.assert_called_once_with(retrieval.CROSS_ENCODER_MODEL)
        finally:
            retrieval._cross_encoder = existing_encoder

    def test_rerank_candidates_are_capped_by_setting(self):
        dense = [
            {"chunk_id": "dense-1", "text": "first", "dense_score": 1.0},
            {"chunk_id": "dense-2", "text": "second", "dense_score": 0.9},
        ]
        bm25 = [
            {"chunk_id": "bm25-1", "text": "third", "bm25_score": 3.0},
        ]
        with patch.object(retrieval, "RERANK_CANDIDATE_LIMIT", 2), patch.object(
            retrieval, "dense_search", return_value=dense
        ), patch.object(
            retrieval, "bm25_search", return_value=bm25
        ), patch.object(
            retrieval,
            "rerank",
            side_effect=lambda query, candidates, top_n: candidates[:top_n],
        ) as rerank:
            retrieval.retrieve("question", top_n_final=1)

        self.assertEqual(len(rerank.call_args.args[1]), 2)


if __name__ == "__main__":
    unittest.main()
