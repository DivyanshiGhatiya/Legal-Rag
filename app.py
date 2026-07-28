# app.py
"""
Week 1 + Week 2 Legal RAG prototype:
- Sidebar with title/description.
- PDF upload (multiple) → validation → extraction → cleaning → preview → save.
- Chat interface powered by Grok LLM (unchanged from Week 1).
"""

import os
import streamlit as st
from utils.grok_client import get_grok_response
from utils.pdf_processor import (
    validate_pdf,
    extract_text_and_metadata,
    clean_text,
    save_processed,
)

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Legal RAG – Week 2",
    page_icon="⚖️",
    layout="centered",
)

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("⚖️ Legal RAG")
    st.caption(
        "Week 2 prototype: PDF upload → ingestion → preview.\n"
        "Week 1 chat with Grok LLM remains at the bottom."
    )
    st.info(
        "Uploaded PDFs are saved to `data/uploads/` (raw). "
        "Processed JSONs go to `processed_data/`."
    )

# ----------------------------------------------------------------------
# Ensure folders exist
# ----------------------------------------------------------------------
UPLOAD_DIR = os.path.join("data", "uploads")
PROCESSED_DIR = os.path.join("processed_data")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Main area: PDF uploader & processing
# ----------------------------------------------------------------------
st.header("📄 Upload Legal PDF(s) for Processing")

uploaded_files = st.file_uploader(
    label="Drop one or more PDF files here",
    type=["pdf"],
    accept_multiple_files=True,
    help="Each PDF will be validated, text extracted, cleaned, and saved as JSON.",
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.expander(f"📄 {uploaded_file.name}", expanded=True):
            # Read the file bytes once
            file_bytes = uploaded_file.getvalue()

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

            # Augment metadata with original filename (used for saving)
            extraction["file_name"] = uploaded_file.name

            # ---------- Cleaning ----------
            cleaned_text = clean_text(extraction["text"])

            # ---------- Preview ----------
            st.subheader("📑 Metadata")
            st.json(
                {
                    "File name": extraction["file_name"],
                    "Number of pages": extraction["num_pages"],
                    "File size (bytes)": extraction["file_size"],
                }
            )

            st.subheader("📝 Cleaned Text Preview")
            # Show first 2000 characters with a scrollable area; full text is saved.
            preview = cleaned_text if len(cleaned_text) <= 2000 else cleaned_text[:2000] + "..."
            st.text_area(
                label="Preview (first 2000 chars)",
                value=preview,
                height=200,
                disabled=True,
            )

            # ---------- Save ----------
            try:
                saved_path = save_processed(
                    metadata=extraction,
                    cleaned_text=cleaned_text,
                    output_dir=PROCESSED_DIR,
                )
                st.success(f"✅ Processed JSON saved to: `{saved_path}`")
            except Exception as exc:
                st.error(f"❌ Failed to save processed output: {exc}")

# ----------------------------------------------------------------------
# Chat interface (unchanged from Week 1)
# ----------------------------------------------------------------------
st.header("💬 Ask a Legal Question")

# Initialize chat history in session state if not present
if "messages" not in st.session_state:
    st.session_state.messages = []  # each entry: {"role": "user"/"assistant", "content": str}

# Display prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box at the bottom
if prompt := st.chat_input("Type your legal question…"):
    # 1️⃣ Add user message to history & display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2️⃣ Get assistant response from Grok
    with st.chat_message("assistant"):
        with st.spinner("Thinking with Grok…"):
            # Optional system hint to steer the model
            legal_prompt = (
                "You are a helpful legal assistant. Answer concisely and cite "
                "general legal principles unless the user provides specific documents.\n\n"
                f"User question: {prompt}"
            )
            answer = get_grok_response(legal_prompt)
            st.markdown(answer)

    # 3️⃣ Append assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ----------------------------------------------------------------------
# Footer (optional)
# ----------------------------------------------------------------------
st.divider()
st.caption(
    "Built with Streamlit + Grok API. "
    "Week 2 adds PDF ingestion; Week 3 will add chunking & embeddings."
)
