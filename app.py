# app.py
"""
Legal RAG prototype focused on embedding and vector storage:
- Sidebar with title/description.
- PDF upload (multiple) → validation → extraction → cleaning → saving.
- Hierarchical chunking of the cleaned text → preview + metadata → saved JSON.
- Local embedding with Ollama and storage in ChromaDB.
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
from utils.vector_store import (
    compute_file_hash,
    document_already_embedded,
    embed_chunks,
    save_index_manifest,
)

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Legal RAG – Week 3",
    page_icon="⚖️",
    layout="centered",
)

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("⚖️ Legal RAG")
    st.caption(
        "Prototype: PDF upload → ingestion → hierarchical chunking → embedding → ChromaDB storage."
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

            # ---------- Save processed JSON (Week 2 output) ----------
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

            # ---------- Hierarchical Chunking (Week 3) ----------
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

                # ----- UPDATED METADATA DISPLAY -----
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
                # ------------------------------------

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
                        collection_name="legal_rag_chunks",
                        persist_dir=os.path.join("data", "chroma"),
                    )
                    if already_embedded:
                        st.info("ℹ️ This document was already embedded; reusing the existing ChromaDB vectors.")
                    else:
                        embedded_records = embed_chunks(
                            chunks,
                            collection_name="legal_rag_chunks",
                            persist_dir=os.path.join("data", "chroma"),
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
                        else:
                            st.info("ℹ️ No chunks were embedded.")
                except Exception as exc:
                    st.warning(f"⚠️ Embedding could not be completed: {exc}")

# ----------------------------------------------------------------------
# Embedding + vector-store status
# ----------------------------------------------------------------------
st.header("🗂️ Embedding + Vector Store")
st.info("This version focuses on PDF ingestion, chunking, embedding, and storing vectors in ChromaDB.")

st.header("💬 Ask a Legal Question")
prompt = st.text_area(
    "Type your legal question",
    placeholder="Example: What are the key terms of this contract?",
    height=120,
)
if st.button("Submit"):
    if prompt and prompt.strip():
        st.info("This version currently focuses on document processing and vector storage.")
    else:
        st.warning("Please enter a question first.")
