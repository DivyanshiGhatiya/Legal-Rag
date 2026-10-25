"""Backend controller for the Legal RAG application.

The Streamlit frontend imports this module as its application API. Pipeline
execution and backend error handling stay here so the UI only renders results.
"""

import json
import logging
import os
import time
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
from explainability import (
    build_llm_verifiers,
    compute_claim_attribution,
    generate_evidence_report,
)
from utils.generation import generate_answer, user_friendly_model_error
from utils.pdf_processor import (
    clean_text,
    extract_text_and_metadata,
    save_processed,
    validate_pdf,
)
from utils.query_rewriter import looks_like_follow_up, rewrite_query
from utils.retrieval import refresh_corpus_cache, retrieve


UPLOAD_DIR = os.path.join("data", "uploads")
PROCESSED_DIR = "processed_data"
CHUNKED_DIR = "chunked_data"
INDEX_MANIFEST_PATH = os.path.join("data", "chroma_manifest.json")
logger = logging.getLogger(__name__)


def _error_result(message: str, stage: str, **extra: Any) -> Dict[str, Any]:
    """Return the stable failure shape consumed by the frontend."""
    return {
        **extra,
        "success": False,
        "error": {"stage": stage, "message": message},
    }


def build_explainability_data(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    citation_report: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build optional explainability results without affecting a generated answer."""
    if not str(answer or "").strip():
        return None

    claim_verifier = None
    contradiction_verifier = None
    explainability_error = None
    if retrieved_chunks:
        verifier_setup_started = time.perf_counter()
        try:
            claim_verifier, contradiction_verifier = build_llm_verifiers()
        except Exception as exc:
            logger.exception(
                "pipeline_stage=verifier_setup status=failed error_type=%s",
                type(exc).__name__,
            )
            explainability_error = user_friendly_model_error(exc)
        finally:
            logger.info(
                "pipeline_stage=verifier_setup duration_seconds=%.3f",
                time.perf_counter() - verifier_setup_started,
            )

    try:
        claim_verification_started = time.perf_counter()
        claim_attributions = compute_claim_attribution(
            answer,
            retrieved_chunks,
            verifier=claim_verifier,
        )
        logger.info(
            "pipeline_stage=claim_verification duration_seconds=%.3f claims=%d",
            time.perf_counter() - claim_verification_started,
            len(claim_attributions),
        )
        evidence_report_started = time.perf_counter()
        evidence_report = generate_evidence_report(
            answer,
            retrieved_chunks,
            claim_attributions=claim_attributions,
            retrieval_metadata=retrieved_chunks,
            contradiction_verifier=contradiction_verifier,
            citation_report=citation_report,
        )
        logger.info(
            "pipeline_stage=evidence_report duration_seconds=%.3f",
            time.perf_counter() - evidence_report_started,
        )
    except Exception as exc:
        logger.exception(
            "pipeline_stage=explainability_analysis status=failed error_type=%s",
            type(exc).__name__,
        )
        message = (
            f"Explainability processing failed ({type(exc).__name__}). "
            "The answer and retrieved evidence remain available; check server logs."
        )
        return {
            "citation_report": citation_report,
            "claim_attributions": [],
            "evidence_report": None,
            "error": message,
        }

    return {
        "citation_report": citation_report,
        "claim_attributions": claim_attributions,
        "evidence_report": evidence_report,
        "error": explainability_error,
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
    query_started = time.perf_counter()
    base = {"query": query, "retrieval_query": query, "hits": [], "answer": None}
    if not query or not query.strip():
        logger.info(
            "pipeline_stage=query_validation duration_seconds=%.3f",
            time.perf_counter() - query_started,
        )
        logger.info(
            "pipeline_stage=backend_query_total duration_seconds=%.3f",
            time.perf_counter() - query_started,
        )
        return _error_result("Question cannot be empty.", "query_validation", **base)

    retrieval_query = query
    if looks_like_follow_up(query, conversation_history or []):
        rewrite_started = time.perf_counter()
        retrieval_query = rewrite_query(query, conversation_history or [])
        logger.info(
            "pipeline_stage=query_rewrite_total duration_seconds=%.3f changed=%s",
            time.perf_counter() - rewrite_started,
            retrieval_query != query,
        )
    base["retrieval_query"] = retrieval_query

    try:
        retrieval_started = time.perf_counter()
        hits = retrieve(
            query=retrieval_query,
            collection_name=DEFAULT_COLLECTION_NAME,
            persist_dir=DEFAULT_PERSIST_DIR,
            document_filter=document_filter,
            section_filter=section_filter,
            top_n_final=top_n_final,
            use_reranker=use_reranker,
        )
        base["hits"] = hits
        logger.info(
            "pipeline_stage=backend_retrieval duration_seconds=%.3f chunks=%d",
            time.perf_counter() - retrieval_started,
            len(hits),
        )
    except Exception as exc:
        logger.exception(
            "pipeline_stage=backend_retrieval status=failed error_type=%s",
            type(exc).__name__,
        )
        logger.info(
            "pipeline_stage=backend_query_total duration_seconds=%.3f",
            time.perf_counter() - query_started,
        )
        return _error_result(str(exc), "retrieval", **base)

    if not base["hits"]:
        base["success"] = True
        base["answer"] = {
            "answer": "No relevant context was retrieved for this question from the currently indexed documents.",
            "used_chunk_ids": [],
        }
        logger.info(
            "pipeline_stage=backend_query_total duration_seconds=%.3f",
            time.perf_counter() - query_started,
        )
        return base

    try:
        # Generation receives history only to resolve references; retrieval stays document-grounded.
        generation_started = time.perf_counter()
        base["answer"] = generate_answer(
            query,
            base["hits"],
            conversation_history=conversation_history,
        )
        logger.info(
            "pipeline_stage=backend_answer_generation duration_seconds=%.3f",
            time.perf_counter() - generation_started,
        )
    except Exception as exc:
        logger.exception(
            "pipeline_stage=backend_answer_generation status=failed error_type=%s",
            type(exc).__name__,
        )
        logger.info(
            "pipeline_stage=backend_query_total duration_seconds=%.3f",
            time.perf_counter() - query_started,
        )
        return _error_result(user_friendly_model_error(exc), "generation", **base)

    base["success"] = True
    logger.info(
        "pipeline_stage=backend_query_total duration_seconds=%.3f",
        time.perf_counter() - query_started,
    )
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
