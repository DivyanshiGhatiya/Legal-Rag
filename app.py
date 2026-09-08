"""Backend controller for the Legal RAG application.

The Streamlit frontend imports this module as its application API. Pipeline
execution and backend error handling stay here so the UI only renders results.
"""

import json
import os
from typing import Any, Dict, List, Optional

from utils.chunker import create_chunks_hierarchical, save_chunked
from utils.embedding import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIR,
    compute_file_hash,
    document_already_embedded,
    embed_chunks,
    get_collection,
    save_index_manifest,
)
from utils.generation import generate_answer
from utils.pdf_processor import (
    clean_text,
    extract_text_and_metadata,
    save_processed,
    validate_pdf,
)
from utils.retrieval import refresh_corpus_cache, retrieve


UPLOAD_DIR = os.path.join("data", "uploads")
PROCESSED_DIR = "processed_data"
CHUNKED_DIR = "chunked_data"
INDEX_MANIFEST_PATH = os.path.join("data", "chroma_manifest.json")


def _error_result(message: str, stage: str, **extra: Any) -> Dict[str, Any]:
    """Return the stable failure shape consumed by the frontend."""
    return {
        **extra,
        "success": False,
        "error": {"stage": stage, "message": message},
    }


def process_document(file_name: str, file_bytes: bytes) -> Dict[str, Any]:
    """Validate, persist, chunk, and embed one uploaded PDF."""
    result: Dict[str, Any] = {
        "success": False,
        "error": None,
        "file_name": file_name,
        "document_hash": compute_file_hash(file_bytes),
        "extraction": None,
        "cleaned_text": "",
        "chunks": [],
        "paths": {},
        "embedded_count": 0,
        "already_embedded": False,
    }

    is_valid, validation_message = validate_pdf(file_bytes)
    if not is_valid:
        return _error_result(validation_message, "validation", **result)

    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        raw_path = os.path.join(UPLOAD_DIR, os.path.basename(file_name))
        with open(raw_path, "wb") as output_file:
            output_file.write(file_bytes)
        result["paths"]["raw"] = raw_path
    except Exception as exc:
        return _error_result(str(exc), "upload_save", **result)

    try:
        extraction = extract_text_and_metadata(file_bytes)
        extraction["file_name"] = file_name
        result["extraction"] = extraction
    except Exception as exc:
        return _error_result(str(exc), "extraction", **result)

    try:
        cleaned_text = clean_text(extraction["text"])
        result["cleaned_text"] = cleaned_text
        result["paths"]["processed"] = save_processed(
            metadata=extraction,
            cleaned_text=cleaned_text,
            output_dir=PROCESSED_DIR,
        )
    except Exception as exc:
        return _error_result(str(exc), "cleaning_or_processed_save", **result)

    try:
        chunks = create_chunks_hierarchical(
            doc_meta=extraction,
            cleaned_text=cleaned_text,
            chunk_size=1000,
            overlap=200,
        )
        result["chunks"] = chunks
        if chunks:
            result["paths"]["chunked"] = save_chunked(chunks, output_dir=CHUNKED_DIR)
    except Exception as exc:
        return _error_result(str(exc), "chunking", **result)

    if not chunks:
        result["success"] = True
        return result

    try:
        already_embedded = document_already_embedded(
            file_name,
            result["document_hash"],
            collection_name=DEFAULT_COLLECTION_NAME,
            persist_dir=DEFAULT_PERSIST_DIR,
        )
        result["already_embedded"] = already_embedded
        if not already_embedded:
            embedded_records = embed_chunks(
                chunks,
                collection_name=DEFAULT_COLLECTION_NAME,
                persist_dir=DEFAULT_PERSIST_DIR,
                document_name=file_name,
                document_hash=result["document_hash"],
            )
            result["embedded_count"] = len(embedded_records)
            if embedded_records:
                result["paths"]["index_manifest"] = save_index_manifest(
                    embedded_records,
                    output_path=INDEX_MANIFEST_PATH,
                )
                refresh_corpus_cache(
                    collection_name=DEFAULT_COLLECTION_NAME,
                    persist_dir=DEFAULT_PERSIST_DIR,
                )
    except Exception as exc:
        return _error_result(str(exc), "embedding", **result)

    result["success"] = True
    return result


def query_legal_documents(
    query: str,
    document_filter: Optional[str] = None,
    section_filter: Optional[str] = None,
    top_n_final: int = 5,
    use_reranker: bool = True,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Run retrieval and answer generation, returning one structured result."""
    base = {"query": query, "hits": [], "answer": None}
    if not query or not query.strip():
        return _error_result("Question cannot be empty.", "query_validation", **base)

    try:
        hits = retrieve(
            query=query,
            collection_name=DEFAULT_COLLECTION_NAME,
            persist_dir=DEFAULT_PERSIST_DIR,
            document_filter=document_filter,
            section_filter=section_filter,
            top_n_final=top_n_final,
            use_reranker=use_reranker,
        )
        base["hits"] = hits
    except Exception as exc:
        return _error_result(str(exc), "retrieval", **base)

    if not base["hits"]:
        base["success"] = True
        base["answer"] = {
            "answer": "No relevant context was retrieved for this question from the currently indexed documents.",
            "used_chunk_ids": [],
        }
        return base

    try:
        # Generation receives history only to resolve references; retrieval stays document-grounded.
        base["answer"] = generate_answer(
            query,
            base["hits"],
            conversation_history=conversation_history,
        )
    except Exception as exc:
        return _error_result(str(exc), "generation", **base)

    base["success"] = True
    return base


def list_indexed_documents() -> List[str]:
    """Return unique source names currently stored in ChromaDB."""
    try:
        collection = get_collection(DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR)
        metadata = collection.get(include=["metadatas"]).get("metadatas") or []
        return sorted(
            {
                item.get("source_name")
                for item in metadata
                if item and item.get("source_name")
            }
        )
    except Exception:
        return []


def remove_indexed_document(document_name: str) -> Dict[str, Any]:
    """Delete every indexed chunk for a source document and refresh retrieval caches."""
    if not document_name:
        return {"success": False, "deleted_count": 0, "error": "Document name is required."}

    try:
        collection = get_collection(DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR)
        matching = collection.get(
            where={"source_name": document_name},
            include=["metadatas"],
        )
        chunk_ids = matching.get("ids") or []
        if chunk_ids:
            collection.delete(ids=chunk_ids)

        manifest_path = os.path.join(DEFAULT_PERSIST_DIR, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            documents = manifest.get("documents", []) if isinstance(manifest, dict) else manifest
            remaining = [
                entry for entry in documents
                if entry.get("source_name") != document_name
            ]
            with open(manifest_path, "w", encoding="utf-8") as manifest_file:
                json.dump({"documents": remaining}, manifest_file, indent=2, ensure_ascii=False)

        refresh_corpus_cache(
            collection_name=DEFAULT_COLLECTION_NAME,
            persist_dir=DEFAULT_PERSIST_DIR,
        )
        return {"success": True, "deleted_count": len(chunk_ids), "error": None}
    except Exception as exc:
        return {"success": False, "deleted_count": 0, "error": str(exc)}


def get_corpus_metrics() -> Dict[str, Any]:
    """Return lightweight corpus metrics for dashboard cards."""
    try:
        collection = get_collection(DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR)
        return {
            "indexed_documents": len(list_indexed_documents()),
            "indexed_chunks": collection.count(),
            "error": None,
        }
    except Exception as exc:
        return {"indexed_documents": 0, "indexed_chunks": 0, "error": str(exc)}
