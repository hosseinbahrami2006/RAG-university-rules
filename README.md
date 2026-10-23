# Local University Rules RAG Assistant

A local Retrieval-Augmented Generation (RAG) application that answers questions about undergraduate academic regulations using a Markdown document.

The application uses Hugging Face models, LangChain components, and a persistent Chroma vector database. It provides command-line tools for indexing the document and asking questions.

> **Disclaimer:** `university_rules.md` describes the fictional Northbridge University and is intended for teaching purposes. It is not an official source of university policy.

## Features

- Loads academic regulations from a Markdown file.
- Removes content beginning at a recognized `Part 13` heading, if present.
- Splits the document into overlapping text chunks.
- Generates embeddings with a Sentence Transformers model.
- Stores embeddings and document metadata in a persistent Chroma database.
- Retrieves relevant chunks for each question.
- Generates answers using a locally loaded Hugging Face language model.
- Includes source filenames and chunk IDs for traceability.
- Prompts the model to acknowledge insufficient evidence instead of inventing policies.

## Project Structure

```text
.
├── rag_agent_local.py
├── university_rules.md
├── requirements.txt
├── README.md
└── chroma_db/             # Created during indexing
```

## Technology Stack

| Component | Technology |
| --- | --- |
| Language | Python |
| Document processing | LangChain Text Splitters |
| Embeddings | Sentence Transformers |
| Vector database | Chroma |
| Model loading and inference | Hugging Face Transformers |
| RAG composition | LangChain |
| Default language model | `Qwen/Qwen2.5-3B-Instruct` |
| Default embedding model | `sentence-transformers/all-MiniLM-L6-v2` |

This implementation does not use Ollama, LangGraph, or tool calling. Despite the script's name, it implements a retrieval-and-generation pipeline rather than a tool-using agent.

## Requirements

- Python 3.11 recommended.
- Internet access for initial package installation and model downloads.
- Sufficient disk space for model files and the vector database.
- Enough RAM or GPU memory to load the selected language model.

A compatible GPU is recommended for faster generation. CPU inference is possible but may be slow. Device placement is handled through `device_map="auto"`.

The default 3B model requires several gigabytes of memory. Actual requirements depend on model precision, device placement, and input length.

## Installation

### 1. Create a virtual environment

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU acceleration, install the appropriate PyTorch build for your system using the instructions at:

https://pytorch.org/get-started/locally/

## Quick Start

Run commands from the project directory.

### 1. Index the document

```bash
python rag_agent_local.py index
```

This command:

1. Reads `university_rules.md`.
2. Applies the optional `Part 13` cutoff.
3. Splits the document into chunks.
4. Downloads or loads the embedding model.
5. Replaces the contents of the configured Chroma collection.
6. Stores the new chunks and embeddings.

> **Important:** Reindexing deletes the existing configured collection before rebuilding it. Other collections are not intentionally reset.

### 2. Start the chat

```bash
python rag_agent_local.py chat
```

The language model is downloaded on first use if it is not already cached.

Example questions:

```text
What is the maximum course load for a student on academic probation?
How many days do I have to appeal a final grade?
When are AI tools allowed in coursework?
What are the graduation requirements?
```

Type `exit` or `quit` to end the session.

If the database is empty, the application asks you to run the indexing command first.

## Configuration

Configuration is read from environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOC_PATH` | `university_rules.md` beside the script | Markdown document to index |
| `PERSIST_DIR` | `chroma_db` beside the script | Persistent database directory |
| `COLLECTION_NAME` | `university_rules` | Chroma collection name |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `LLM_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Language model |
| `TOP_K` | `4` | Number of chunks retrieved per question |

`TOP_K` should be a positive integer.

### Linux / macOS example

```bash
export DOC_PATH="./university_rules.md"
export PERSIST_DIR="./chroma_db"
export TOP_K="4"
export LLM_MODEL="Qwen/Qwen2.5-3B-Instruct"

python rag_agent_local.py index
python rag_agent_local.py chat
```

### Windows PowerShell example

```powershell
$env:DOC_PATH = "./university_rules.md"
$env:PERSIST_DIR = "./chroma_db"
$env:TOP_K = "4"
$env:LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"

python rag_agent_local.py index
python rag_agent_local.py chat
```

An `.env` file is not loaded automatically by this script.

### Using local model directories

`LLM_MODEL` and `EMBED_MODEL` can point to compatible local Hugging Face model directories.

```bash
export LLM_MODEL="/path/to/local/qwen-model"
export EMBED_MODEL="/path/to/local/embedding-model"
```

An Ollama model identifier such as `qwen2.5:3b` is not supported by this implementation.

### Chunking settings

The following settings are defined directly in `rag_agent_local.py`:

```python
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
```

These values represent character counts, not token counts.

Reindex after changing the document, chunking settings, or embedding model. Use the same embedding model during indexing and chat.

## How It Works

```text
Markdown document
    ↓
Preprocessing
    ↓
Overlapping text chunks
    ↓
Embedding model
    ↓
Persistent Chroma database

User question
    ↓
Similarity retrieval
    ↓
Relevant chunks + prompt
    ↓
Local language model
    ↓
Answer with source references
```

Each indexed chunk includes:

- `source`: the document filename.
- `chunk_id`: a sequential chunk identifier.

Source references use this format:

```text
Sources: [source=university_rules.md, chunk_id=2]
```

Chunk IDs may change after reindexing.

## Document Scope

The supplied document covers sections 1–11, including:

- Enrollment and registration.
- Course load and attendance.
- Examinations and grading.
- Academic standing and probation.
- Assignments and academic integrity.
- Leave, withdrawal, graduation, and grade appeals.

The preprocessing function recognizes line-start headings such as `Part 13` or `## PART 13` and removes that heading and everything after it.

The supplied document has no such heading, so it is indexed in full. A heading written only as `13. ...` does not trigger this cutoff.

## Limitations

- Each question is processed independently; conversation history is not retained.
- Retrieval uses similarity search without a relevance-score threshold.
- The insufficient-evidence response is prompt-based and is not guaranteed.
- Source references identify retrieved chunks; they do not independently verify every generated claim.
- Small language models may produce inaccurate answers or inconsistent citations.
- The current generation pipeline may include the input prompt in its output. To suppress prompt echoing, add `return_full_text=False` to the `pipeline(...)` configuration.
- `temperature=0.4` does not enable sampling because `do_sample=False`.
- This is an educational prototype, not an authoritative academic advising system.

## Troubleshooting

### Document not found

Check that `university_rules.md` exists beside the script, or set `DOC_PATH` to the correct file path.

### Empty vector database

Run:

```bash
python rag_agent_local.py index
```

Ensure that indexing and chat use the same database directory and collection name.

### Model download fails

Check your internet connection and available disk space. For offline use, download both models beforehand and configure local model paths.

### Out-of-memory errors or slow generation

Use a smaller compatible model, reduce `TOP_K`, or use hardware with more available memory.

### Errors after changing the embedding model

Rebuild the index with the new embedding model. If incompatible collection settings persist, use a new collection name or database directory and index again.

## Local Execution and Privacy

Embedding generation, retrieval, and language-model inference run locally. The script does not send questions to a hosted language-model API.

Initial model downloads require network access unless the models are already available locally. The vector database stores document chunks and metadata on disk.

## Dependency Reproducibility

`requirements.txt` lists the required packages with minimum versions and a Transformers major-version constraint. It is not a fully pinned lock file.

After validating the application in your environment, you can record exact installed versions:

```bash
python -m pip freeze > requirements-lock.txt
```
