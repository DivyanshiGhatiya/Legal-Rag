# app.py
"""
Week 1 Legal RAG prototype:
- Sidebar with title/description.
- PDF upload widget (UI only).
- Chat interface powered by Grok LLM.
"""

import os
import streamlit as st
from utils.grok_client import get_grok_response

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Legal RAG – Week 1",
    page_icon="⚖️",
    layout="centered",
)

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("⚖️ Legal RAG")
    st.caption(
        "Week 1 prototype: PDF upload (UI only) + chat with Grok LLM.\n"
        "Future weeks will add text extraction, embeddings, and retrieval."
    )
    # Optional: show where uploads would go
    st.info("Uploaded PDFs would be saved to `data/uploads/` (not processed yet).")

# ----------------------------------------------------------------------
# Ensure upload directory exists (creates it if missing)
# ----------------------------------------------------------------------
UPLOAD_DIR = os.path.join("data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Main area: PDF uploader
# ----------------------------------------------------------------------
st.header("📄 Upload a Legal PDF (UI only)")
uploaded_file = st.file_uploader(
    label="Drop a PDF here or click to browse",
    type=["pdf"],
    accept_multiple_files=False,
    help="The file is saved to disk but not read or processed in Week 1."
)

if uploaded_file is not None:
    # Save the uploaded file to the uploads folder (purely for demonstration)
    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Saved `{uploaded_file.name}` to `{UPLOAD_DIR}`.")
else:
    st.caption("No file uploaded yet.")

# ----------------------------------------------------------------------
# Chat interface
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
            # Optionally prepend a system hint to guide the model:
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
    "Week 1 focuses on UI and LLM integration; PDF processing arrives in Week 2."
)
