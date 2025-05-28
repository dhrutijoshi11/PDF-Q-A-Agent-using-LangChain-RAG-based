# PDF Q&A Agent
📄 An interactive PDF question-answering web app using a Retrieval-Augmented Generation (RAG) pipeline powered by LangChain and OpenAI GPT-4.

## Overview
PDF Q&A Agent allows users to upload or select local PDF documents and ask natural language questions about their content. The system uses:

- PDF loading and chunking
- Embeddings for semantic search (OpenAI embeddings + FAISS vector store)
- A Retrieval QA chain powered by GPT-4 via LangChain
The app is built with Streamlit for an interactive, chat-style UI.

## Features
- Upload or use local PDF files for Q&A
- Text chunking with overlap for better context retrieval
- Embedding-based semantic search via FAISS
- GPT-4 powered answers with source document referencing
- Persistent chat history during the session
- Source document snippets shown for transparency

## Tech Stack

- Python 3.13
- Streamlit — Web application framework
- LangChain — RAG pipeline management
- OpenAI GPT-4 — Large Language Model API
- FAISS — Vector similarity search
- PyPDFLoader — PDF document loader from `langchain_community`

## Installation
Clone the repository:

```bash
git clone https://github.com/yourusername/pdf-qa-agent.git
cd pdf-qa-agent
```

Create a Python virtual environment and activate it:

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

(Optional) Create a .env file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```
## Usage
Run the Streamlit app:

```bash
streamlit run app.py
```

- Upload a PDF file or use the sample local PDF (`data/your_pdf_file.pdf`)
- Ask questions in natural language via the chat interface
- View answers along with the source text chunks

## Project Structure

```bash
pdf-qa-agent/
├── data/
│   └── pdf files (place your local PDFs here)
├── utils/
│   └── rag_pipeline.py      # Loaders, embeddings, retriever pipeline
├── app.py                  # Streamlit frontend app
├── requirements.txt        # Python dependencies
├── .env                    # API keys (optional)
```

## How It Works
- `rag_pipeline.py` loads a PDF, splits it into overlapping chunks for context
- Creates vector embeddings using OpenAIEmbeddings
- Stores vectors in FAISS for similarity search
- Builds a RetrievalQA chain with GPT-4 as the language model
- `app.py` provides the Streamlit UI for uploading/selecting PDFs and chatting with the agent

## Notes
- Requires an OpenAI API key for GPT-4 access.
- Chunk size and overlap in the text splitter can be adjusted for different PDF sizes.
- For large PDFs, indexing may take some time initially.

