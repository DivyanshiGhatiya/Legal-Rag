<<<<<<< Updated upstream

"""
Legal RAG prototype:
- Sidebar with title/description.
- PDF upload (multiple) → validation → extraction → cleaning → saving.
- Hierarchical chunking of the cleaned text → preview + metadata → saved JSON.
- Local embedding with Ollama and storage in ChromaDB.
- Hybrid retrieval (dense + BM25 + RRF fusion + cross-encoder rerank) demo.
=======
"""Backend controller for the Legal RAG application.
The Streamlit frontend imports this module as its application API. Keeping
the pipeline here makes ingestion and question answering reusable from other
frontends and gives the UI one consistent error/result contract.
>>>>>>> Stashed changes
"""

import os
from typing import Any, Dict, List, Optional

from utils.chunker import create_chunks_hierarchical, save_chunked
from utils.pdf_processor import (
    validate_pdf,
    extract_text_and_metadata,
    clean_text,
    save_processed,
)
from utils.embedding import (
    compute_file_hash,
    document_already_embedded,
    embed_chunks,
    save_index_manifest,
    get_collection,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIR,
)
from utils.retrieval import refresh_corpus_cache, retrieve
from utils.generation import generate_answer
from utils.query_rewriter import rewrite_query

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
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
    """Validate, persist, chunk, and embed one uploaded PDF.

    All pipeline work belongs here; the returned dictionary is safe for a UI
    to render even when an individual stage fails.
    """
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
        base["answer"] = generate_answer(query, base["hits"])
    except Exception as exc:
        return _error_result(str(exc), "generation", **base)

<<<<<<< Updated upstream
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if st.button("🧹 New conversation"):
    st.session_state.chat_messages = []
    st.rerun()

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Try to list already-indexed documents for the dropdown
available_docs = []
try:
    if os.path.isdir(PERSIST_DIR):
        coll = get_collection(collection_name=COLLECTION, persist_dir=PERSIST_DIR)
        all_meta = coll.get(include=["metadatas"])
        available_docs = sorted({
            m.get("source_name")
            for m in (all_meta.get("metadatas") or [])
            if m and m.get("source_name")
        })
except Exception:
    available_docs = []
=======
    base["success"] = True
    return base
>>>>>>> Stashed changes


def list_indexed_documents() -> List[str]:
    """Return unique source names currently stored in ChromaDB."""
    try:
        collection = get_collection(DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR)
        metadata = collection.get(include=["metadatas"]).get("metadatas") or []
        return sorted({item.get("source_name") for item in metadata if item and item.get("source_name")})
    except Exception:
        return []


<<<<<<< Updated upstream
prompt = st.text_area(
    "Type your legal question",
    placeholder="Example: What are the termination clauses in this contract?",
    height=120,
)

if st.button("🔍 Search"):
    if not prompt or not prompt.strip():
        st.warning("Please enter a question first.")
    else:
        conversation_history = list(st.session_state.chat_messages)
        retrieval_query = rewrite_query(prompt, conversation_history)
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        with st.spinner("🔎 Running hybrid retrieval (dense + BM25 + RRF + rerank)…"):
            try:
                hits = retrieve(
                    query=retrieval_query,
                    collection_name=COLLECTION,
                    persist_dir=PERSIST_DIR,
                    document_filter=doc_filter,
                    section_filter=section_filter,
                    top_n_final=top_n_final,
                    use_reranker=use_reranker,
                )
            except Exception as exc:
                st.error(f"❌ Retrieval failed: {exc}")
                hits = []

        if not hits:
            st.warning("No matching chunks found. Try uploading and embedding a document first.")
        else:
            st.success(f"✅ Found {len(hits)} relevant chunks")

            # ---------- Generation (Grok) ----------
            with st.spinner("🧠 Generating answer from retrieved chunks…"):
                try:
                    gen_result = generate_answer(
                        prompt,
                        hits,
                        conversation_history=conversation_history,
                    )
                except Exception as exc:
                    gen_result = None
                    st.error(f"❌ Answer generation failed: {exc}")

            if gen_result:
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": gen_result["answer"]}
                )
                st.subheader("📝 Answer")
                st.markdown(gen_result["answer"])
                citation_report = gen_result.get("citation_report")
                if citation_report:
                    if citation_report["is_valid"]:
                        st.success("✅ Citation verification passed")
                    else:
                        st.warning("⚠️ Citation verification found issues")
                        for problem in citation_report["problems"]:
                            st.caption(problem)
                    if citation_report["citations"]:
                        st.caption(
                            "Verified chunks: "
                            + ", ".join(f"`{cid}`" for cid in citation_report["citations"])
                        )
                st.markdown("---")

            st.subheader("🔍 Retrieved Chunks")
            for rank, hit in enumerate(hits, start=1):
                score_label = (
                    f"Rerank: {hit['rerank_score']:.3f}"
                    if hit.get("rerank_score") is not None
                    else f"RRF: {hit['rrf_score']:.4f}"
                )
                with st.expander(
                    f"#{rank}  •  {score_label}  •  📄 {hit['document_name']}"
                ):
                    st.caption(f"**Chunk ID:** `{hit['chunk_id']}`")
                    st.caption(f"**Section:** {hit['section'] or '(none)'} / {hit['subsection'] or '(none)'}")
                    st.caption(f"**Hierarchy:** {hit['hierarchy'] or '(none)'}")
                    st.caption(
                        f"**Scores** — dense: {hit['dense_score']:.3f}"
                        if hit.get("dense_score") is not None else "**Scores** — dense: n/a"
                    )
                    st.markdown("---")
                    st.markdown(hit["text"])
=======
def get_corpus_metrics() -> Dict[str, Any]:
    """Return lightweight corpus metrics for dashboard cards."""
    try:
        collection = get_collection(DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR)
        count = collection.count()
        return {"indexed_documents": len(list_indexed_documents()), "indexed_chunks": count, "error": None}
    except Exception as exc:
        return {"indexed_documents": 0, "indexed_chunks": 0, "error": str(exc)}
>>>>>>> Stashed changes
