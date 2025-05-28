import streamlit as st
import tempfile
from pathlib import Path
from utils.rag_pipline import load_pdf_and_build_qa_chain
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

st.set_page_config(page_title="📄 PDF Q&A Agent", layout="centered")
st.title("📄 PDF Q&A Agent (Chat Style)")
st.sidebar.markdown("""
### 🧠 Tech Stack
- 🐍 **Python 3.13**
- 📄 **Streamlit** – Web UI
- ⚙️ **LangChain** – RAG pipeline
- 🧠 **OpenAI GPT-4** – LLM via API
- 🔍 **FAISS** – Vector store for similarity search
- 📚 **PDF Loader** – `PyPDFLoader` from `langchain_community`
""")


# Chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File selection
use_local_file = st.checkbox("Use local PDF file instead of upload?", value=False)

if use_local_file:
    pdf_path = "data/your_pdf_file.pdf"
else:
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    pdf_path = None
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

# Load and build QA chain
if pdf_path:
    if (
        "qa_chain" not in st.session_state
        or st.session_state.get("file_name") != pdf_path
    ):
        with st.spinner("🔧 Indexing PDF..."):
            st.session_state.qa_chain = load_pdf_and_build_qa_chain(pdf_path)
            st.session_state.file_name = pdf_path
            st.session_state.chat_history = []  # Reset chat when new PDF is loaded
        st.success("✅ PDF ready for questions!")

# Show full chat history persistently
if "qa_chain" in st.session_state:
    chat_container = st.container()

    with chat_container:
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                st.markdown(a)

    # Input must come after history render
    user_input = st.chat_input("Ask a question about the PDF")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("🤔 Thinking…"):
            result = st.session_state.qa_chain(user_input)
            answer = result["result"]
            sources = result["source_documents"]

        # Save to chat history first
        st.session_state.chat_history.append((user_input, answer))

        # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Show sources
        with st.expander("🔍 Sources"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**Chunk {i}** (p. {doc.metadata.get('page', '?')})")
                st.write(doc.page_content)
                st.markdown("---")
else:
    st.info("⬆️ Upload a PDF or select local file to begin.")
