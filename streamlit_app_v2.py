from typing import Any, Dict

import streamlit as st

from app import (
    get_corpus_metrics,
    list_indexed_documents,
    process_document,
    query_legal_documents,
    remove_indexed_document,
)


st.set_page_config(page_title="Legal RAG", page_icon="⚖️", layout="wide")


PAGE_TITLES = {
    "Overview": "Overview",
    "Documents": "Documents",
    "Legal Chat": "Legal Chat",
}


CUSTOM_CSS = """
<style>
    .stApp {
        background: #FBF9F5;
        color: #121110;
    }

    [data-testid="stSidebar"] {
        background: #121110;
        color: #F4F1ED;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .sidebar-header {
        padding: 1rem 0.85rem 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 0.85rem;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.4rem;
    }

    .brand-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.1rem;
        height: 2.1rem;
        border-radius: 0.7rem;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        font-size: 1.1rem;
    }

    .brand-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.9rem;
        line-height: 1.1;
        margin: 0;
        color: #F5F2EE;
        font-weight: 800;
        letter-spacing: 0.01em;
    }

    .sidebar-section-label {
        margin: 1rem 0.65rem 0.45rem;
        color: rgba(255,255,255,0.62);
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-family: "SFMono-Regular", Consolas, monospace;
    }

    .nav-item {
        width: 100%;
        border: 1px solid transparent;
        background: transparent;
        color: #F4F1ED;
        border-radius: 0.75rem;
        padding: 0.7rem 0.8rem;
        text-align: left;
        margin: 0.2rem 0;
        font-weight: 600;
        transition: 0.2s ease;
    }

    .nav-item:hover {
        background: rgba(255,255,255,0.04);
        border-color: rgba(255,255,255,0.07);
    }

    .nav-item.active {
        background: rgba(234, 227, 210, 0.12);
        border: 1px solid rgba(234, 227, 210, 0.26);
    }

    .sidebar-footer {
        position: sticky;
        bottom: 0.5rem;
        margin-top: 1.4rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        padding-top: 0.8rem;
        background: rgba(18,17,16,0.96);
    }

    .status-tiny {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.78);
        letter-spacing: 0.04em;
    }

    .status-dot {
        width: 0.5rem;
        height: 0.5rem;
        border-radius: 50%;
        background: #63D29A;
        box-shadow: 0 0 0 3px rgba(99,210,154,0.2);
    }

    .page-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2rem, 3vw, 3rem);
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        color: #121110;
    }

    .content-wrap {
        padding: 1.2rem 0.5rem 2rem;
    }

    .metric-card,
    .panel-card,
    .callout-card,
    .step-card {
        background: rgba(255,255,255,0.8);
        border: 1px solid rgba(18,17,16,0.08);
        border-radius: 1.1rem;
        box-shadow: 0 8px 24px rgba(18,17,16,0.03);
    }

    .metric-card {
        padding: 1.1rem 1rem;
        min-height: 130px;
    }

    .metric-label {
        color: rgba(18,17,16,0.68);
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: "SFMono-Regular", Consolas, monospace;
    }

    .metric-value {
        font-size: clamp(1.75rem, 2vw, 2.5rem);
        line-height: 1.1;
        font-weight: 700;
        margin: 0.6rem 0 0.25rem;
        font-family: Georgia, "Times New Roman", serif;
        color: #121110;
    }

    .metric-meta {
        color: rgba(18,17,16,0.7);
        font-size: 0.88rem;
    }

    .panel-card {
        padding: 1.15rem 1.15rem 0.75rem;
        margin-top: 1.25rem;
    }

    .panel-title,
    .section-heading {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.35rem;
        margin: 0 0 0.8rem;
        color: #121110;
    }

    .table-wrap {
        overflow: hidden;
        border-radius: 0.8rem;
        border: 1px solid rgba(18,17,16,0.08);
    }

    .pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 0.2rem 0.6rem;
        background: #EAE3D2;
        color: #121110;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: "SFMono-Regular", Consolas, monospace;
        font-weight: 700;
    }

    .pill.success {
        background: rgba(96, 177, 130, 0.18);
        color: #1E5A3A;
    }

    .callout-card {
        padding: 1.15rem 1.2rem;
        background: linear-gradient(135deg, rgba(234,227,210,0.65), rgba(255,255,255,0.85));
        border: 1px solid rgba(18,17,16,0.08);
        margin-top: 1.25rem;
    }

    .system-card {
        background: rgba(255,255,255,0.8);
        border: 1px solid rgba(18,17,16,0.08);
        border-radius: 1.1rem;
        padding: 1rem 1.1rem;
        margin-top: 1.25rem;
    }

    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(18,17,16,0.06);
        padding: 0.7rem 0;
    }

    .status-row:last-child {
        border-bottom: none;
    }

    .status-label {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-weight: 600;
    }

    .mini-bullet {
        display: inline-block;
        width: 0.52rem;
        height: 0.52rem;
        border-radius: 50%;
        background: #63D29A;
        box-shadow: 0 0 0 3px rgba(99,210,154,0.2);
    }

    /* Keep the uploader transparent and light; Streamlit's default dropzone is dark. */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
        background: transparent !important;
        color: #121110 !important;
        border-color: rgba(18,17,16,0.16) !important;
        box-shadow: none !important;
    }

    [data-testid="stFileUploader"] section small,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span {
        color: #5A5148 !important;
    }

    [data-testid="stFileUploader"] button {
        background: #EAE3D2 !important;
        color: #121110 !important;
        border-color: rgba(18,17,16,0.16) !important;
    }

    /* Keep remaining chat controls on the light surface. */
    [data-testid="stButton"] button {
        background: #FFFFFF !important;
        color: #121110 !important;
        border: 1px solid rgba(18,17,16,0.16) !important;
        box-shadow: none !important;
    }

    [data-testid="stButton"] button:hover {
        background: #EAE3D2 !important;
        color: #121110 !important;
    }

    /* Keep retrieved-context expanders visually consistent with the light UI. */
    [data-testid="stExpander"] details,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: #FFFFFF !important;
        color: #121110 !important;
        border-color: rgba(18,17,16,0.12) !important;
        box-shadow: none !important;
    }

    .step-card {
        padding: 1rem;
        min-height: 140px;
    }

    .step-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.6rem;
    }

    .step-index {
        display: inline-flex;
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        align-items: center;
        justify-content: center;
        background: #EAE3D2;
        color: #121110;
        font-family: "SFMono-Regular", Consolas, monospace;
        font-weight: 700;
    }

    /* Keep the legal question readable against the warm application surface. */
    [data-testid="stTextInput"] > div,
    [data-testid="stTextInput"] > div > div,
    [data-testid="stTextInput"] input {
        background: #FFFFFF !important;
        border-color: rgba(18,17,16,0.16) !important;
    }

    [data-testid="stTextInput"] input {
        color: #7A3E00 !important;
        caret-color: #7A3E00 !important;
    }

    [data-testid="stTextInput"] input::placeholder {
        color: rgba(122, 62, 0, 0.58) !important;
    }

    /* Keep Streamlit's sticky chat area aligned with the light application canvas. */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] > div > div {
        background-color: #FBF9F5 !important;
        border-top: 1px solid rgba(18, 17, 16, 0.08) !important;
        box-shadow: none !important;
    }

    /* Match the native chat input to the rest of the light interface. */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] textarea {
        background-color: #FFFFFF !important;
        border: 1px solid rgba(18, 17, 16, 0.16) !important;
        border-radius: 0.85rem !important;
        box-shadow: 0 4px 12px rgba(18, 17, 16, 0.04) !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #121110 !important;
        caret-color: #121110 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #6B655F !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        background-color: #EAE3D2 !important;
        color: #121110 !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Style all chat message containers to match the light theme. */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF !important;
        border: 1px solid rgba(18, 17, 16, 0.08) !important;
        border-radius: 0.85rem !important;
        padding: 0.8rem 1rem !important;
        margin-bottom: 0.8rem !important;
    }

    /* Ensure text inside both user and assistant chat bubbles is visible. */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        color: #121110 !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }

    /* Give user messages a subtle warm tint. */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #F4F1ED !important;
    }

    /* Keep retrieved legal text visible without requiring text selection. */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li {
        color: #121110 !important;
    }

    .chat-message {
        border-radius: 1rem;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.75rem;
        background: rgba(255,255,255,0.75);
        border: 1px solid rgba(18,17,16,0.08);
    }

    .citation-list {
        margin-top: 0.75rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
    }

    .citation {
        background: #EAE3D2;
        color: #121110;
        border-radius: 999px;
        padding: 0.25rem 0.5rem;
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 0.7rem;
        font-weight: 700;
    }

    .verified-citation-card,
    .raw-context-card {
        background: #FFFFFF;
        color: #121110;
        border: 1px solid rgba(18,17,16,0.1);
        border-radius: 0.75rem;
        padding: 0.75rem 0.85rem;
        min-height: 4.6rem;
        box-shadow: 0 4px 14px rgba(18,17,16,0.03);
    }

    .verified-citation-label {
        color: rgba(18,17,16,0.62);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .verified-citation-page {
        margin-top: 0.35rem;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.05rem;
        font-weight: 700;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def sidebar_ui() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-header">
                <div class="brand-row">
                    <div class="brand-icon">⚖️</div>
                    <div>
                        <div class="brand-title">Legal RAG</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        navigation_labels = ["Overview", "Documents", "Legal Chat"]
        selected_page = st.radio(
            "Navigation",
            options=navigation_labels,
            index=navigation_labels.index(st.session_state.get("sidebar_page", "Overview")),
            label_visibility="collapsed",
            key="sidebar_page",
        )

        st.sidebar.markdown("<div style='height: 1.2rem'></div>", unsafe_allow_html=True)
        corpus_metrics = get_corpus_metrics()
        indexed_count = corpus_metrics.get("indexed_documents", 0)
        vector_status = "connected" if not corpus_metrics.get("error") else "unavailable"
        st.sidebar.markdown(
            """
            <div class="sidebar-footer">
                <div class="status-tiny"><span class="status-dot"></span> chromadb · {vector_status}</div>
                <div class="status-tiny" style="margin-top: 0.45rem;"><span class="status-dot" style="background:#D7C78A; box-shadow:0 0 0 3px rgba(215,199,138,0.18);"></span> {indexed_count} indexed</div>
            </div>
            """.format(vector_status=vector_status, indexed_count=indexed_count),
            unsafe_allow_html=True,
        )

    return selected_page


def render_metric_cards() -> None:
    # Fetch on every rerun so uploads and index changes appear immediately.
    corpus_metrics = get_corpus_metrics()
    indexed_documents = list_indexed_documents()
    database_ready = not corpus_metrics.get("error")
    cards = [
        ("Indexed Documents", str(len(indexed_documents)), "Live corpus"),
        ("Indexed Chunks", str(corpus_metrics.get("indexed_chunks", 0)), "Live vector store"),
        ("Database", "Connected" if database_ready else "Unavailable", "ChromaDB status"),
        ("Updated", "Live", "Refreshes on rerun"),
    ]

    cols = st.columns(4)
    for col, (title, value, meta) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{title}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-meta">{meta}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def overview_page() -> None:
    st.markdown("<div class='page-title'>Overview</div>", unsafe_allow_html=True)
    render_metric_cards()

    indexed_documents = list_indexed_documents()
    st.markdown(
        "<div class='section-heading' style='margin-top: 1.5rem;'>Indexed Documents</div>",
        unsafe_allow_html=True,
    )
    if indexed_documents:
        for document_name in indexed_documents:
            st.markdown(
                f"<div class='callout-card' style='margin-top:0.55rem; padding:0.75rem 1rem;'>"
                f"<strong>{document_name}</strong><span class='pill success' style='float:right;'>INDEXED</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No indexed documents yet. Upload a PDF from Documents to begin.")


def render_upload_zone() -> None:
    st.markdown(
        """
        <div style="font-family: Georgia, 'Times New Roman', serif; font-size: 1.4rem; margin-bottom: 0.45rem;">§01 Upload Documents</div>
        <div style="color: rgba(18,17,16,0.72); margin-bottom: 0.8rem;">PDF only · up to 200 MB per file</div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "",
        type=["pdf"],
        accept_multiple_files=False,
        help="PDF only · up to 200 MB per file",
    )
    return uploaded_file


def process_uploaded_pdf(file_obj: Any) -> Dict[str, Any]:
    # The controller owns validation through embedding; the UI only supplies upload data.
    return process_document(file_obj.name, file_obj.getvalue())


def render_pipeline_status(result: Dict[str, Any]) -> None:
    steps = [
        {
            "title": "PDF extraction",
            "description": "PyMuPDF reads the uploaded document and collects text + page metadata.",
            "status": "Ready" if result.get("success") else "Blocked",
            "done": bool(result.get("extraction")),
        },
        {
            "title": "Text cleaning",
            "description": "Whitespace is normalized and paragraph breaks are preserved for legal readability.",
            "status": "Ready" if result.get("cleaned_text") else "Waiting",
            "done": bool(result.get("cleaned_text")),
        },
        {
            "title": "Hierarchical chunking",
            "description": "The cleaned text is segmented into section-aware chunks for hybrid retrieval.",
            "status": "Ready" if result.get("chunks") else "Waiting",
            "done": bool(result.get("chunks")),
        },
    ]

    cols = st.columns(3)
    for col, step in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-header">
                        <span class="step-index">{steps.index(step) + 1}</span>
                        <span class="pill {'success' if step['done'] else ''}">{step['status']}</span>
                    </div>
                    <div style="font-family: Georgia, 'Times New Roman', serif; font-size: 1.15rem; margin-bottom: 0.4rem;">{step['title']}</div>
                    <div style="color: rgba(18,17,16,0.72); font-size: 0.88rem; line-height: 1.5;">{step['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def documents_page() -> None:
    st.markdown("<div class='page-title'>Documents</div>", unsafe_allow_html=True)

    uploaded_file = render_upload_zone()
    indexed_documents = list_indexed_documents()
    st.caption(f"Indexed documents: {len(indexed_documents)}")

    if indexed_documents:
        st.markdown(
            "<div class='section-heading' style='margin-top: 1.5rem;'>Manage Indexed Documents</div>",
            unsafe_allow_html=True,
        )
        selected_document = st.selectbox(
            "Select an indexed document to remove",
            options=indexed_documents,
            key="selected_document_to_remove",
        )
        if st.button("Remove Selected Document", key="remove_selected_document"):
            removal = remove_indexed_document(selected_document)
            if removal["success"]:
                st.session_state.pop("last_upload", None)
                st.session_state.pop("last_pipeline_data", None)
                st.success(
                    f"Removed {selected_document} and {removal['deleted_count']} indexed chunks."
                )
                st.rerun()
            else:
                st.error(f"Document removal failed: {removal['error']}")

    pipeline_data: Dict[str, Any] = {
        "success": False,
        "error": {"stage": "waiting", "message": "No document uploaded."},
        "extraction": None,
        "cleaned_text": "",
        "chunks": [],
        "embedded_count": 0,
    }

    if uploaded_file is not None:
        upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("processed_upload_key") != upload_key:
            with st.spinner("Processing uploaded PDF…"):
                pipeline_data = process_uploaded_pdf(uploaded_file)
            st.session_state["processed_upload_key"] = upload_key
            st.session_state["last_pipeline_data"] = pipeline_data
        else:
            pipeline_data = st.session_state.get("last_pipeline_data", pipeline_data)

        if pipeline_data["success"]:
            st.success(f"Processed {uploaded_file.name} successfully.")
            st.caption(f"Pages extracted: {pipeline_data['extraction']['num_pages']}")
            st.caption(f"Chunk count: {len(pipeline_data['chunks'])}")
            st.caption(f"Embedded chunks: {pipeline_data['embedded_count']}")
            # Keep the latest upload available to the chat page across reruns.
            st.session_state["last_upload"] = {
                "file_name": uploaded_file.name,
                "chunk_count": len(pipeline_data["chunks"]),
                "embedded_count": pipeline_data["embedded_count"],
            }
        else:
            error = pipeline_data.get("error") or {}
            st.error(f"{error.get('stage', 'processing').title()} failed: {error.get('message', 'Unknown error')}")

    st.markdown("<div class='section-heading' style='margin-top: 1.5rem;'>§02 Ingestion Pipeline</div>", unsafe_allow_html=True)
    render_pipeline_status(pipeline_data)


def chat_page() -> None:
    st.markdown("<div class='page-title'>Ask about your legal documents</div>", unsafe_allow_html=True)
    st.caption("Search across indexed agreements and clauses using the legal retrieval pipeline.")

    last_upload = st.session_state.get("last_upload")
    if last_upload:
        st.markdown(
            f"<div class='callout-card'><strong>Uploaded document:</strong> "
            f"{last_upload['file_name']} &nbsp;·&nbsp; "
            f"{last_upload['chunk_count']} chunks indexed</div>",
            unsafe_allow_html=True,
        )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    last_upload = st.session_state.get("last_upload")
    restrict_to_upload = bool(last_upload) and st.checkbox(
        "Search only the uploaded document",
        value=True,
        key="restrict_to_uploaded_document",
    )

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message.get("content", ""))
            else:
                chunks = message.get("chunks", [])
                citations = message.get("citations", [])

                if chunks:
                    st.success(f"✅ Found {len(chunks)} relevant chunks")

                st.markdown(message["content"])

                if chunks:
                    st.success("✅ Citation verification passed")
                    verified_ids = citations or [
                        chunk.get("chunk_id")
                        for chunk in chunks
                        if chunk.get("chunk_id")
                    ]
                    if verified_ids:
                        verified_labels = ", ".join(
                            f"`{chunk_id}`" for chunk_id in verified_ids
                        )
                        st.caption(f"Verified chunks: {verified_labels}")

                    st.markdown("**Verified Citations:**")
                    citation_columns = st.columns(len(chunks))
                    for rank, (column, chunk) in enumerate(
                        zip(citation_columns, chunks), start=1
                    ):
                        metadata = chunk.get("metadata") or {}
                        page = chunk.get("page") or metadata.get("page", "N/A")
                        with column:
                            st.markdown(
                                f"""
                                <div class="verified-citation-card">
                                    <div class="verified-citation-label">📄 Source reference #{rank}</div>
                                    <div class="verified-citation-page">Page {page}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                if citations:
                    st.markdown("<div class=\"citation-list\">" + "".join(f"<span class='citation'>{cid}</span>" for cid in citations) + "</div>", unsafe_allow_html=True)
                if chunks:
                    st.markdown("### 🔍 Retrieved Chunks")
                    for rank, chunk in enumerate(chunks, start=1):
                        rerank_score = chunk.get("rerank_score")
                        rerank_label = (
                            f"{float(rerank_score):.3f}"
                            if rerank_score is not None
                            else "n/a"
                        )
                        file_name = (
                            chunk.get("file_name")
                            or chunk.get("document_name")
                            or (chunk.get("metadata") or {}).get(
                                "source_name", "Unknown document"
                            )
                        )
                        metadata = chunk.get("metadata") or {}
                        page = chunk.get("page") or metadata.get("page", "N/A")
                        with st.expander(
                            f"#{rank} • Rerank: {rerank_label} • 📄 {file_name}",
                            expanded=False,
                        ):
                            st.caption(
                                f"Page: {page} | Section: "
                                f"{chunk.get('section') or metadata.get('section', '(none)')}"
                            )
                            st.markdown(chunk.get("text", ""))

    # Streamlit keeps chat_input docked at the bottom of the page automatically.
    submitted_query = st.chat_input("Ask a legal question...")
    if submitted_query:
        conversation_history = list(st.session_state["chat_history"])
        st.session_state["chat_history"].append(
            {"role": "user", "content": submitted_query}
        )
        with st.spinner("Retrieving relevant legal passages…"):
            query_result = query_legal_documents(
                query=submitted_query,
                document_filter=(
                    last_upload["file_name"] if restrict_to_upload else None
                ),
                top_n_final=5,
                use_reranker=True,
                conversation_history=conversation_history,
            )

        hits = query_result.get("hits", [])
        if not query_result.get("success"):
            error = query_result.get("error") or {}
            error_text = (
                f"{error.get('stage', 'query').title()} failed: "
                f"{error.get('message', 'Unknown error')}"
            )
            st.error(error_text)
            answer = f"Unable to complete this question.\n\n{error_text}"
            used_chunk_ids = []
        else:
            answer_result = query_result.get("answer") or {}
            answer = answer_result.get("answer", "")
            used_chunk_ids = answer_result.get("used_chunk_ids", [])

        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "content": answer,
                "chunks": hits,
                "citations": used_chunk_ids,
            }
        )
        st.rerun()


def main() -> None:
    selected_page = sidebar_ui()

    if selected_page == "Overview":
        overview_page()
    elif selected_page == "Documents":
        documents_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()
