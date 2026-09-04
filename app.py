
"""
Legal RAG prototype:
- Sidebar with title/description.
- PDF upload (multiple) → validation → extraction → cleaning → saving.
- Hierarchical chunking of the cleaned text → preview + metadata → saved JSON.
- Local embedding with Ollama and storage in ChromaDB.
- Hybrid retrieval (dense + BM25 + RRF fusion + cross-encoder rerank) demo.
"""

import os
import streamlit as st
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
from utils.retrieval import retrieve, refresh_corpus_cache
from utils.generation import generate_answer
from utils.query_rewriter import rewrite_query

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Legal RAG – Week 3",
    page_icon="⚖️",
    layout="centered",
)

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("⚖️ Legal RAG")
    st.caption(
        "Prototype: PDF upload → ingestion → hierarchical chunking → embedding → "
        "ChromaDB storage → hybrid retrieval (dense + BM25 + rerank)."
    )
    st.info(
        "Uploaded PDFs are saved to `data/uploads/` (raw). "
        "Processed JSONs go to `processed_data/`. "
        "Chunked JSONs go to `chunked_data/`."
    )

# ----------------------------------------------------------------------
# Ensure folders exist
# ----------------------------------------------------------------------
UPLOAD_DIR = os.path.join("data", "uploads")
PROCESSED_DIR = os.path.join("processed_data")
CHUNKED_DIR = os.path.join("chunked_data")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHUNKED_DIR, exist_ok=True)

COLLECTION = DEFAULT_COLLECTION_NAME
PERSIST_DIR = DEFAULT_PERSIST_DIR

# ----------------------------------------------------------------------
# Main area: PDF uploader & processing
# ----------------------------------------------------------------------
st.header("📄 Upload Legal PDF(s) for Processing")

uploaded_files = st.file_uploader(
    label="Drop one or more PDF files here",
    type=["pdf"],
    accept_multiple_files=True,
    help="Each PDF will be validated, text extracted, cleaned, saved as JSON, then hierarchically chunked.",
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.expander(f"📄 {uploaded_file.name}", expanded=True):
            # Read the file bytes once
            file_bytes = uploaded_file.getvalue()
            document_hash = compute_file_hash(file_bytes)

            # ---------- Validation ----------
            is_valid, err_msg = validate_pdf(file_bytes)
            if not is_valid:
                st.error(f"❌ Validation failed: {err_msg}")
                continue

            # Save raw upload (for later weeks)
            raw_save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(raw_save_path, "wb") as f:
                f.write(file_bytes)
            st.caption(f"Raw file saved to `{raw_save_path}`")

            # ---------- Extraction ----------
            try:
                extraction = extract_text_and_metadata(file_bytes)
            except Exception as exc:
                st.error(f"❌ Text extraction failed: {exc}")
                continue

            # Augment metadata with original filename (used for saving & chunking)
            extraction["file_name"] = uploaded_file.name

            # ---------- Cleaning ----------
            cleaned_text = clean_text(extraction["text"])

            # ---------- Save processed JSON ----------
            try:
                saved_processed_path = save_processed(
                    metadata=extraction,
                    cleaned_text=cleaned_text,
                    output_dir=PROCESSED_DIR,
                )
                st.success(f"✅ Processed JSON saved to: `{saved_processed_path}`")
            except Exception as exc:
                st.error(f"❌ Failed to save processed output: {exc}")
                continue  # Skip chunking if we couldn't save the base file

            # ---------- Hierarchical Chunking ----------
            try:
                chunks = create_chunks_hierarchical(
                    doc_meta=extraction,
                    cleaned_text=cleaned_text,
                    chunk_size=1000,   # feel free to tune
                    overlap=200,
                )
            except Exception as exc:
                st.error(f"❌ Chunking failed: {exc}")
                continue

            # Show chunking results
            st.subheader(f"🔢 Chunks Created: {len(chunks)}")
            if chunks:
                # Preview of the first chunk
                first_preview = chunks[0]["text"]
                if len(first_preview) > 500:
                    first_preview = first_preview[:500] + "..."
                st.text_area(
                    label="Preview of first chunk (first 500 chars)",
                    value=first_preview,
                    height=150,
                    disabled=True,
                )

                with st.expander("Show chunk metadata (first 5)"):
                    meta_rows = []
                    for i, c in enumerate(chunks[:5], start=1):
                        hierarchy_path = " > ".join(
                            c.get("metadata", {}).get("hierarchy", [])
                        ) or "(no headings detected)"
                        meta_rows.append(
                            {
                                "Chunk ID": c["chunk_id"],
                                "Chunk #": i,
                                "Length (chars)": c["chunk_length"],
                                "Hierarchy": hierarchy_path,
                            }
                        )
                    st.dataframe(meta_rows)

                # Save chunked JSON
                try:
                    saved_chunk_path = save_chunked(chunks, output_dir=CHUNKED_DIR)
                    st.success(f"✅ Chunked JSON saved to: `{saved_chunk_path}`")
                except Exception as exc:
                    st.error(f"❌ Failed to save chunked output: {exc}")

                # Embed chunks locally with Ollama + ChromaDB
                try:
                    already_embedded = document_already_embedded(
                        uploaded_file.name,
                        document_hash,
                        collection_name=COLLECTION,
                        persist_dir=PERSIST_DIR,
                    )
                    if already_embedded:
                        st.info(
                            "ℹ️ Embedding already exists for this document. "
                            "Skipping re-embedding and reusing the stored vectors."
                        )
                    else:
                        embedded_records = embed_chunks(
                            chunks,
                            collection_name=COLLECTION,
                            persist_dir=PERSIST_DIR,
                            document_name=uploaded_file.name,
                            document_hash=document_hash,
                        )
                        if embedded_records:
                            manifest_path = save_index_manifest(
                                embedded_records,
                                output_path=os.path.join("data", "chroma_manifest.json"),
                            )
                            st.success(
                                f"✅ Embedding complete: {len(embedded_records)} chunks stored locally in ChromaDB."
                            )
                            st.caption(f"Manifest saved to: `{manifest_path}`")
                            # New chunks were added — refresh the in-memory BM25 index
                            refresh_corpus_cache(collection_name=COLLECTION, persist_dir=PERSIST_DIR)
                        else:
                            st.info("ℹ️ No chunks were embedded.")
                except Exception as exc:
                    st.warning(f"⚠️ Embedding could not be completed: {exc}")

# ----------------------------------------------------------------------
# Embedding + vector-store status
# ----------------------------------------------------------------------
st.header("🗂️ Embedding + Vector Store")
st.info(
    "Documents are chunked hierarchically, embedded locally with Ollama "
    "(`nomic-embed-text`), and stored in ChromaDB."
)

# ----------------------------------------------------------------------
# 💬 Ask a Legal Question (Hybrid Retrieval Demo)
# ----------------------------------------------------------------------
st.header("💬 Ask a Legal Question")

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

doc_choice = st.selectbox(
    "Restrict search to a specific document (optional):",
    options=["(All documents)"] + available_docs,
    index=0,
)
doc_filter = None if doc_choice == "(All documents)" else doc_choice

section_filter_input = st.text_input(
    "Restrict search to a section/heading containing (optional):",
    placeholder="e.g. Confidentiality",
)
section_filter = section_filter_input.strip() or None

col1, col2 = st.columns(2)
with col1:
    top_n_final = st.slider("Final number of chunks to return:", 1, 10, 5)
with col2:
    use_reranker = st.checkbox("Use cross-encoder reranking", value=True)

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