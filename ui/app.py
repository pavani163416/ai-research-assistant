import os
import streamlit as st
from services.rag_service import RAGService

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_FOLDER = "documents/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "rag" not in st.session_state:
    st.session_state.rag = RAGService()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chunks" not in st.session_state:
    st.session_state.chunks = 0

rag = st.session_state.rag

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:

    st.title("📚 AI Research Assistant")
    st.markdown("---")

    st.subheader("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        save_path = os.path.join(
            UPLOAD_FOLDER,
            uploaded_file.name
        )

        if st.session_state.document_name != uploaded_file.name:

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Processing document..."):

                result = rag.process_document(save_path)

            st.session_state.document_ready = True
            st.session_state.document_name = uploaded_file.name
            st.session_state.chunks = result["chunks"]

        st.success("✅ Document Ready")

    else:
        st.info("Upload a PDF to begin.")

    st.markdown("---")

    st.subheader("📊 Document Information")

    st.metric(
        "File",
        st.session_state.document_name if st.session_state.document_name else "-"
    )

    st.metric(
        "Chunks",
        st.session_state.chunks
    )

    st.markdown("---")

    st.subheader("⚙️ Components")

    st.success("PDF Loader")
    st.success("Text Splitter")
    st.success("Embeddings")
    st.success("FAISS")
    st.success("Gemini")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------
# Main UI
# --------------------------------------------------

st.title("📚 AI Research Assistant")
st.caption("Chat with your uploaded document using RAG + Gemini")

# Display previous messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        # Display sources only for assistant messages
        if msg["role"] == "assistant":
            
            if "metrics" in msg and msg["metrics"]:
                metrics = msg["metrics"]
                with st.expander("⏱️ Performance Metrics"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Retrieval", f"{metrics.get('retrieval_time', 0):.2f}s")
                    col2.metric("Generation", f"{metrics.get('generation_time', 0):.2f}s")
                    col3.metric("Total", f"{metrics.get('total_time', 0):.2f}s")
                    col4.metric("Chunks", metrics.get("chunks_retrieved", 0))

            if "sources" in msg and msg["sources"]:
                with st.expander("📄 Retrieved Sources"):
                    for source in msg["sources"]:
                        page_str = f"Page {source['page_start']}"
                        if source['page_start'] != source['page_end']:
                            page_str += f"-{source['page_end']}"
                            
                        st.markdown(f"**{page_str} | Chunk ID:** {source['chunk_id']} | **Score:** {source['score']:.4f}")
                        st.write(source["text"])
                        st.markdown("---")

# --------------------------------------------------
# Chat Input
# --------------------------------------------------

prompt = st.chat_input("Ask anything about your uploaded document...")

if prompt:

    if not st.session_state.document_ready:

        st.warning("Please upload a PDF first.")
        st.stop()

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = rag.ask(prompt)

                answer = response.get("answer", "Error generating answer.")
                sources = response.get("sources", [])
                metrics = response.get("metrics", {})

            except Exception as e:

                answer = f"❌ Error: {e}"
                sources = []
                metrics = {}

        st.markdown(answer)

        # Display metrics
        if metrics:
            with st.expander("⏱️ Performance Metrics"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Retrieval", f"{metrics.get('retrieval_time', 0):.2f}s")
                col2.metric("Generation", f"{metrics.get('generation_time', 0):.2f}s")
                col3.metric("Total", f"{metrics.get('total_time', 0):.2f}s")
                col4.metric("Chunks", metrics.get("chunks_retrieved", 0))

        # Display retrieved chunks
        if sources:
            with st.expander("📄 Retrieved Sources"):
                for source in sources:
                    page_str = f"Page {source['page_start']}"
                    if source['page_start'] != source['page_end']:
                        page_str += f"-{source['page_end']}"
                        
                    st.markdown(f"**{page_str} | Chunk ID:** {source['chunk_id']} | **Score:** {source['score']:.4f}")
                    st.write(source["text"])
                    st.markdown("---")

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "metrics": metrics
        }
    )