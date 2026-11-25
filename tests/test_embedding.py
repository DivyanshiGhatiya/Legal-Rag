import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from utils.embedding import embed_chunks


class EmbeddingTests(unittest.TestCase):
    def test_inconsistent_dimensions_fail_before_any_chunk_is_inserted(self):
        client = SimpleNamespace(
            embed=Mock(
                return_value=SimpleNamespace(embeddings=[[0.1, 0.2], [0.3]])
            )
        )
        collection = Mock()

        with (
            patch("utils.embedding.ollama.Client", return_value=client),
            patch("utils.embedding.document_already_embedded", return_value=False),
            patch("utils.embedding.get_collection", return_value=collection),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "inconsistent dimensions: \\[1, 2\\]"
            ):
                embed_chunks(
                    [{"chunk_id": "doc_1", "text": "first"},
                     {"chunk_id": "doc_2", "text": "second"}],
                    document_name="doc.pdf",
                    document_hash="hash",
                )

        collection.add.assert_not_called()

    def test_missing_embedding_for_input_fails_before_insertion(self):
        client = SimpleNamespace(
            embed=Mock(
                return_value=SimpleNamespace(embeddings=[[0.1, 0.2]])
            )
        )
        collection = Mock()

        with (
            patch("utils.embedding.ollama.Client", return_value=client),
            patch("utils.embedding.document_already_embedded", return_value=False),
            patch("utils.embedding.get_collection", return_value=collection),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "1 vectors for 2 inputs"
            ):
                embed_chunks(
                    [{"chunk_id": "doc_1", "text": "first"},
                     {"chunk_id": "doc_2", "text": "second"}],
                    document_name="doc.pdf",
                    document_hash="hash",
                )

        collection.add.assert_not_called()


if __name__ == "__main__":
    unittest.main()
