import json
import os
import tempfile
import unittest
from unittest.mock import patch

import app
from utils.embedding import document_already_embedded


class DocumentIndexingTests(unittest.TestCase):
    def test_same_filename_with_changed_hash_is_not_treated_as_indexed(self):
        with tempfile.TemporaryDirectory() as persist_dir:
            manifest_path = os.path.join(persist_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as manifest_file:
                json.dump(
                    {
                        "documents": [
                            {
                                "source_name": "contract.pdf",
                                "source_hash": "old-hash",
                            }
                        ]
                    },
                    manifest_file,
                )

            self.assertTrue(
                document_already_embedded(
                    "contract.pdf", "old-hash", persist_dir=persist_dir
                )
            )
            self.assertFalse(
                document_already_embedded(
                    "contract.pdf", "new-hash", persist_dir=persist_dir
                )
            )
            self.assertFalse(
                document_already_embedded(
                    "another-name.pdf", "old-hash", persist_dir=persist_dir
                )
            )

    def test_reuploading_changed_file_replaces_old_source_index(self):
        class CollectionStub:
            def get(self, **_kwargs):
                return {
                    "metadatas": [
                        {"source_name": "contract.pdf", "source_hash": "old-hash"}
                    ]
                }

            def count(self):
                return 1

        with tempfile.TemporaryDirectory() as upload_dir:
            with (
                patch.object(app, "UPLOAD_DIR", upload_dir),
                patch.object(app, "compute_file_hash", return_value="new-hash"),
                patch.object(app, "validate_pdf", return_value=(True, "")),
                patch.object(
                    app,
                    "extract_text_and_metadata",
                    return_value={"text": "New contract terms.", "num_pages": 1},
                ),
                patch.object(app, "clean_text", return_value="New contract terms."),
                patch.object(app, "save_processed", return_value="processed.json"),
                patch.object(
                    app,
                    "create_chunks_hierarchical",
                    return_value=[
                        {
                            "chunk_id": "contract_CHUNK_0001",
                            "text": "New contract terms.",
                            "metadata": {},
                        }
                    ],
                ),
                patch.object(app, "save_chunked", return_value="chunks.json"),
                patch.object(app, "get_collection", return_value=CollectionStub()),
                patch.object(
                    app,
                    "remove_indexed_document",
                    return_value={
                        "success": True,
                        "deleted_count": 1,
                        "error": None,
                    },
                ) as remove_document,
                patch.object(app, "document_already_embedded", return_value=False),
                patch.object(
                    app,
                    "embed_chunks",
                    return_value=[{"chunk_id": "contract_CHUNK_0001"}],
                ),
                patch.object(
                    app, "save_index_manifest", return_value="manifest.json"
                ),
                patch.object(app, "refresh_corpus_cache"),
            ):
                result = app.process_document("contract.pdf", b"%PDF-test")

        self.assertTrue(result["success"])
        self.assertEqual(result["embedded_count"], 1)
        remove_document.assert_called_once_with("contract.pdf")


if __name__ == "__main__":
    unittest.main()
