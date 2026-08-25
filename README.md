# OVGU / FIN / Magdeburg Assistant

A multi-agent conversational AI assistant providing information about Otto von Guericke University Magdeburg (OVGU), the Faculty of Informatics (FIN), and the city of Magdeburg. Built with Streamlit, LangGraph, Pydantic AI, and Supabase.

## Features

*   **Conversational Interface:** Ask questions in natural language via a Streamlit web UI, with persistent chat history per session.
*   **Multi-Agent System:** A LangGraph router analyzes each query and dispatches it to one of three specialized agents — OVGU, FIN, or Magdeburg — each with its own domain-scoped system prompt.
*   **Retrieval-Augmented Generation (RAG):** Each agent retrieves the most relevant chunks from a dedicated Supabase/pgvector table via semantic vector search before answering, keeping responses grounded in the ingested source material.
*   **LLM Integration:** Uses OpenAI models — `gpt-4o-mini` (configurable) for response generation and `text-embedding-3-small` for embeddings.
*   **Source Linking:** Formats responses so `(Source: URL)` citations become clickable Markdown links, so answers can be verified against the original page.

## Architecture

```
User Query → Streamlit UI → LangGraph Router
                                   │
                     ┌─────────────┼─────────────┐
                 OVGU Agent    FIN Agent    Magdeburg Agent
                     │             │             │
                     └── retrieve top-k chunks from its own
                         Supabase/pgvector table (RAG) ──┘
                                   │
                          LLM synthesizes response
                                   │
                        Streamlit UI (with source links)
```

1. The user submits a question through the Streamlit chat UI ([streamlit_app.py](streamlit_app.py)).
2. A LangGraph state machine ([graph/agent_graph.py](graph/agent_graph.py)) routes the query to the OVGU, FIN, or Magdeburg agent using keyword-based matching, or falls back to a clarifying message if no domain is clearly matched.
3. The chosen agent ([agents/](agents/)), built with Pydantic AI, embeds the query and calls a Supabase RPC function (`match_ovgu_pages`, `match_fin_pages`, or `match_magdeburg_pages`) to retrieve the top-7 most similar document chunks from its own table.
4. The agent's LLM synthesizes an answer strictly grounded in the retrieved context, citing source URLs.
5. The response flows back through the graph to the UI, where source citations are rendered as clickable links.

The knowledge base itself is populated separately by the ingestion pipeline ([ingestion/ingest_local_data.py](ingestion/ingest_local_data.py)): it crawls URLs listed in the sitemap files under [data/](data/) (or parses linked PDFs), chunks the text, generates embeddings, and stores everything in the corresponding Supabase table defined in [supabase_schema_separate.sql](supabase_schema_separate.sql).

## Tech Stack

*   **Frontend:** Streamlit
*   **Backend/Orchestration:** Python, LangGraph, Pydantic AI
*   **LLM:** OpenAI (`gpt-4o-mini` by default, configurable via `.env`)
*   **Database:** Supabase with the `pgvector` extension — required for both ingestion and retrieval
*   **Data Ingestion:** Crawl4AI for web crawling, `pypdf` for PDF text extraction
*   **Dependencies:** See [requirements.txt](requirements.txt)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
    *   Copy the example environment file: `cp .env.example .env` (or `copy .env.example .env` on Windows).
    *   Edit the `.env` file and add your credentials (see Environment Variables section below).

## Environment Variables

Create a `.env` file in the project root and add the following variables:

```dotenv
# Required — used for embeddings and LLM responses
OPENAI_API_KEY=your_openai_api_key

# Required — the RAG knowledge base is stored in Supabase
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_role_key

# Optional — defaults to gpt-4o-mini if not set
LLM_MODEL=gpt-4o-mini
```

*   Get OpenAI keys from [platform.openai.com](https://platform.openai.com/).
*   Get Supabase URL and Service Key from your Supabase project settings under API.

## Usage

### 1. Set up the Supabase schema

Run [supabase_schema_separate.sql](supabase_schema_separate.sql) against your Supabase project (SQL Editor) to create the `ovgu_pages`, `fin_pages`, and `magdeburg_pages` tables and their vector-search functions.

### 2. Populate the knowledge base

The chatbot answers strictly from retrieved context, so it needs data before it's useful:

```bash
python -m ingestion.ingest_local_data
```

This crawls the URLs listed in the sitemap files under [data/](data/), chunks and embeds the content, and stores it in Supabase.

### 3. Run the assistant

```bash
streamlit run streamlit_app.py
```

Open your browser to the URL Streamlit prints (usually `http://localhost:8501`) and start asking questions about OVGU, FIN, or Magdeburg.

## Project Structure

```
.
├── .env.example                  # Template for required environment variables
├── .gitignore                    # Files ignored by Git (.env, venv/, __pycache__/, ...)
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── streamlit_app.py              # Streamlit chat UI — main application entry point
├── supabase_schema_separate.sql  # pgvector tables + RPC match functions for OVGU/FIN/Magdeburg
├── utils.py                      # Shared OpenAI/Supabase client factories and embedding helper
├── agents/                       # Domain-specific RAG agents (OVGU, FIN, Magdeburg)
├── data/                         # Sitemap XML files used to seed the crawler
├── graph/                        # LangGraph router and orchestration logic
└── ingestion/                    # Web crawler / knowledge-base ingestion pipeline
```