#!/usr/bin/env python3
"""
Local RAG Agent for University Rules
------------------------------------
Commands:
  python3 rag_agent_local.py index
  python3 rag_agent_local.py chat

Environment variables (optional):
  DOC_PATH=./university_rules.md
  PERSIST_DIR=./chroma_db
  COLLECTION_NAME=university_rules
  EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
  LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
  TOP_K=4
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

DOC_PATH = os.getenv("DOC_PATH", str(BASE_DIR / "university_rules.md"))
PERSIST_DIR = os.getenv("PERSIST_DIR", str(BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "university_rules")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
TOP_K = int(os.getenv("TOP_K", "4"))

# Chunking settings
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


@dataclass
class AppContext:
    embeddings: HuggingFaceEmbeddings
    vectorstore: Chroma
    llm: HuggingFacePipeline


# -----------------------------
# Document loading + preprocessing
# -----------------------------
def read_markdown_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document file not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


def cut_from_part_13(text: str) -> str:
    """
    Remove content starting from Part 13 onward.
    Supports patterns like:
      - "Part 13"
      - "PART 13"
      - "## Part 13"
      - "### PART 13"
    """
    pattern = re.compile(r"(?im)^\s{0,3}(#{1,6}\s*)?part\s*13\b.*$")
    match = pattern.search(text)
    if match:
        return text[: match.start()].rstrip()
    return text


def build_documents(raw_text: str, source_name: str) -> List[Document]:
    cleaned = cut_from_part_13(raw_text)
    if not cleaned.strip():
        return []
    return [Document(page_content=cleaned, metadata={"source": source_name})]


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Attach chunk_id for traceable citations
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "unknown_source"

    return chunks


# -----------------------------
# Models
# -----------------------------
def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_llm() -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype="auto",
        device_map="auto",
    )

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=350,
        do_sample=False,
        temperature=0.4,
        repetition_penalty=1.05,
    )

    return HuggingFacePipeline(pipeline=text_gen)


def build_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


# -----------------------------
# Indexing
# -----------------------------
def run_index() -> None:
    print("Starting indexing...")
    print(f"DOC_PATH: {DOC_PATH}")
    print(f"PERSIST_DIR: {PERSIST_DIR}")
    print(f"COLLECTION_NAME: {COLLECTION_NAME}")
    print(f"EMBED_MODEL: {EMBED_MODEL}")

    raw_text = read_markdown_file(DOC_PATH)
    documents = build_documents(raw_text, source_name=Path(DOC_PATH).name)
    print(f"Loaded docs: {len(documents)}")

    if not documents:
        print("No documents after preprocessing (check Part 13 trimming logic).")
        sys.exit(1)

    chunks = split_documents(documents)
    print(f"Chunks: {len(chunks)}")
    if not chunks:
        print("No chunks produced. Check splitter settings.")
        sys.exit(1)

    embeddings = build_embeddings()
    vs = build_vectorstore(embeddings)

    # Reset collection content to avoid duplicates across repeated indexing
    # If you prefer append-only behavior, remove next 2 lines.
    vs.delete_collection()
    vs = build_vectorstore(embeddings)

    vs.add_documents(chunks)

    count = vs._collection.count()
    print(f"Indexed vectors: {count}")

    if count == 0:
        print("Indexing failed: vector count is 0.")
        sys.exit(1)

    print("Indexing completed successfully.")


# -----------------------------
# Retrieval + Answering
# -----------------------------
def format_context(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        source = d.metadata.get("source", "unknown_source")
        chunk_id = d.metadata.get("chunk_id", "unknown_chunk")
        blocks.append(f"[source={source} | chunk_id={chunk_id}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_prompt() -> ChatPromptTemplate:
    template = """
You are a helpful assistant for university students.
Answer ONLY using the provided context.
If the answer is not in the context, say:
"I do not have enough evidence in the provided documents."

Rules:
1) Be concise and accurate.
2) At the end, provide sources as:
   Sources: [source=<name>, chunk_id=<id>], ...
3) Do not invent policies or rules.

Question:
{question}

Context:
{context}
"""
    return ChatPromptTemplate.from_template(template)


def unique_sources(docs: List[Document]) -> List[str]:
    seen = set()
    out = []
    for d in docs:
        source = d.metadata.get("source", "unknown_source")
        chunk_id = d.metadata.get("chunk_id", "unknown_chunk")
        key = (source, chunk_id)
        if key not in seen:
            seen.add(key)
            out.append(f"[source={source}, chunk_id={chunk_id}]")
    return out


def answer_question(ctx: AppContext, question: str) -> str:
    retriever = ctx.vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)

    if not docs:
        return "I do not have enough evidence in the provided documents."

    prompt = build_prompt()
    context_text = format_context(docs)
    chain = prompt | ctx.llm

    response = chain.invoke({"question": question, "context": context_text})

    # HuggingFacePipeline in LangChain returns string output
    answer_text = response if isinstance(response, str) else str(response)

    sources = unique_sources(docs)
    if "Sources:" not in answer_text:
        answer_text = f"{answer_text}\n\nSources: {', '.join(sources)}"

    return answer_text


def run_chat() -> None:
    print("Starting chat mode...")
    print(f"PERSIST_DIR: {PERSIST_DIR}")
    print(f"COLLECTION_NAME: {COLLECTION_NAME}")
    print(f"EMBED_MODEL: {EMBED_MODEL}")
    print(f"LLM_MODEL: {LLM_MODEL}")

    embeddings = build_embeddings()
    vs = build_vectorstore(embeddings)

    count = vs._collection.count()
    print(f"Existing vectors: {count}")
    if count == 0:
        print("Error: Vector database is empty. Run:")
        print("  python3 rag_agent_local.py index")
        sys.exit(1)

    llm = build_llm()
    ctx = AppContext(embeddings=embeddings, vectorstore=vs, llm=llm)

    print("\nChat is ready. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not question:
            continue

        try:
            answer = answer_question(ctx, question)
            print(f"\nAssistant: {answer}\n")
        except KeyboardInterrupt:
            print("\nInterrupted. Bye.")
            break
        except Exception as ex:
            print(f"\nAssistant error: {ex}\n")


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 rag_agent_local.py index")
        print("  python3 rag_agent_local.py chat")
        sys.exit(1)

    command = sys.argv[1].strip().lower()

    try:
        if command == "index":
            run_index()
        elif command == "chat":
            run_chat()
        else:
            print(f"Unknown command: {command}")
            print("Use 'index' or 'chat'.")
            sys.exit(1)
    except FileNotFoundError as ex:
        print(f"File error: {ex}")
        sys.exit(1)
    except Exception as ex:
        print(f"Fatal error: {ex}")
        sys.exit(1)


if __name__ == "__main__":
    main()