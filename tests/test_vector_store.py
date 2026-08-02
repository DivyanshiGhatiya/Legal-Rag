import json
import os
import tempfile
import unittest
from unittest.mock import patch

from utils.vector_store import _get_ollama_client, _normalize_chunk_inputs, document_already_embedded


class VectorStoreTests(unittest.TestCase):
    def test_normalize_chunk_inputs_creates_ids_and_texts(self):
        chunks = ["alpha", {"text": "beta", "chunk_id": "custom-id"}]

        normalized = _normalize_chunk_inputs(chunks)

        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0]["chunk_id"], "chunk_001")
        self.assertEqual(normalized[0]["text"], "alpha")
        self.assertEqual(normalized[1]["chunk_id"], "custom-id")
        self.assertEqual(normalized[1]["text"], "beta")

    def test_get_ollama_client_uses_localhost(self):
        with patch("utils.vector_store.ollama.Client") as mock_client:
            _get_ollama_client()

        mock_client.assert_called_once_with(host="http://127.0.0.1:11434")

    def test_document_already_embedded_checks_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = os.path.join(temp_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "documents": [
                            {
                                "source_name": "dup.pdf",
                                "source_hash": "abc123",
                                "chunk_ids": ["chunk_001"],
                                "records": [],
                            }
                        ]
                    },
                    handle,
                )

            self.assertTrue(
                document_already_embedded(
                    "dup.pdf",
                    "abc123",
                    persist_dir=temp_dir,
                    manifest_path=manifest_path,
                )
            )


if __name__ == "__main__":
    unittest.main()
