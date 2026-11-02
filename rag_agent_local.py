#!/usr/bin/env python3
"""
Local RAG Agent for University Rules (LangChain + LangGraph edition)
--------------------------------------------------------------------
Commands:
  python3 rag_agent_local.py index   # build the vector database
  python3 rag_agent_local.py chat    # chat with the RAG graph
  python3 rag_agent_local.py graph   # print and save the graph diagram (Mermaid)
  python3 rag_agent_local.py web     # serve the browser lab

Environment variables (optional):
  DOC_PATH=./university_rules.md
  PERSIST_DIR=./chroma_db
  COLLECTION_NAME=university_rules
  EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
  LLM_BACKEND=huggingface            # "huggingface", "ollama", or "lmstudio"
  LLM_MODEL=Qwen/Qwen3.5-4B          # use "qwen3.5:4b" with ollama, "qwen3.5-4b" with lmstudio
  TOP_K=4
  SCORE_THRESHOLD=0.25
  CHUNK_SIZE=600
  CHUNK_OVERLAP=100
  MAX_NEW_TOKENS=350                 # raised automatically for Qwen3.5 thinking models
  SHOW_STEPS=0                       # set to 1 to print each graph node as it runs
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langgraph.graph import END, START, StateGraph


# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

DOC_PATH = os.getenv("DOC_PATH", str(BASE_DIR / "university_rules.md"))
PERSIST_DIR = os.getenv("PERSIST_DIR", str(BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "university_rules")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

LLM_BACKEND = os.getenv("LLM_BACKEND", "huggingface").strip().lower()
DEFAULT_LLM_MODELS = {
    "ollama": "qwen3.5:4b",
    "lmstudio": "qwen3.5-4b",
    "huggingface": "Qwen/Qwen3.5-4B",
}
DEFAULT_LLM_MODEL = DEFAULT_LLM_MODELS.get(LLM_BACKEND, "Qwen/Qwen3.5-4B")
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "350"))
# Qwen3.5 thinks by default; short budgets can be spent entirely on reasoning.
# RAG prompts often need ~1000 reasoning tokens before any visible answer.
# Prefer LM Studio context >= 8192 when using this budget.
QWEN35_MIN_NEW_TOKENS = int(os.getenv("QWEN35_MIN_NEW_TOKENS", "2048"))


def is_qwen35(model: str) -> bool:
    normalized = model.lower().replace("_", "").replace("-", "").replace("/", "").replace(":", "").replace(".", "")
    return "qwen35" in normalized


def generation_token_budget(model: str = LLM_MODEL, requested: int = MAX_NEW_TOKENS) -> int:
    if is_qwen35(model):
        return max(requested, QWEN35_MIN_NEW_TOKENS)
    return requested

TOP_K = int(os.getenv("TOP_K", "4"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.25"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

SHOW_STEPS = os.getenv("SHOW_STEPS", "0").strip() == "1"
GRAPH_FILE = BASE_DIR / "rag_graph.mmd"

# Markdown headers used to split the document into logical sections
HEADERS_TO_SPLIT_ON = [
    ("#", "title"),
    ("##", "section"),
    ("###", "subsection"),
]

NO_EVIDENCE_MESSAGE = "I do not have enough evidence in the provided documents."


# -----------------------------
# Document loading + splitting
# -----------------------------
def load_documents(path: str) -> List[Document]:
    """Load the Markdown file with a LangChain document loader."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    loader = TextLoader(str(file_path), encoding="utf-8")
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = file_path.name

    return [doc for doc in documents if doc.page_content.strip()]


def split_documents(documents: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """
    Two-stage splitting:
      1) Split by Markdown headers so each chunk stays inside one section.
      2) Split long sections into overlapping character chunks.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    section_docs: List[Document] = []
    for doc in documents:
        for section in header_splitter.split_text(doc.page_content):
            section.metadata = {**doc.metadata, **section.metadata}
            section_docs.append(section)

    chunks = char_splitter.split_documents(section_docs)

    # Attach chunk_id for traceable citations
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata.setdefault("source", "unknown_source")

    return chunks


# -----------------------------
# Models + vector store
# -----------------------------
def build_embeddings(model_name: str = EMBED_MODEL) -> HuggingFaceEmbeddings:
    # Normalized vectors + cosine distance give meaningful relevance scores (0..1)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(embeddings: HuggingFaceEmbeddings, collection_name: str = COLLECTION_NAME, persist_dir: str = PERSIST_DIR) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_retriever(vectorstore: Chroma) -> VectorStoreRetriever:
    # Chunks below the threshold are dropped, so off-topic questions get no context
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": TOP_K, "score_threshold": SCORE_THRESHOLD},
    )


def build_llm(backend: str = LLM_BACKEND, model: str = LLM_MODEL, base_url: str = "http://127.0.0.1:1234/v1") -> BaseChatModel:
    """
    Return a LangChain *chat* model. Chat models apply Qwen's chat template,
    which is essential for getting synthesized answers instead of copied text.

    Qwen3.5 enables thinking by default. Backends that support it disable thinking
    so RAG answers are not starved by reasoning tokens.
    """
    max_new_tokens = generation_token_budget(model)

    if backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=0.1,
            num_predict=max_new_tokens,
            num_ctx=4096,
            # Disable Qwen3.5 / thinking-model reasoning for direct answers.
            reasoning=False,
        )

    if backend == "huggingface":
        from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        class ChatHuggingFaceNoThink(ChatHuggingFace):
            """Apply chat templates with Qwen3.5 thinking disabled when supported."""

            def _to_chat_prompt(self, messages: list[BaseMessage]) -> str:
                if not messages:
                    raise ValueError("At least one HumanMessage must be provided!")
                if not isinstance(messages[-1], HumanMessage):
                    raise ValueError("Last message must be a HumanMessage!")

                messages_dicts = [self._to_chatml_format(m) for m in messages]
                template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                }
                try:
                    return self.tokenizer.apply_chat_template(messages_dicts, **template_kwargs)
                except TypeError:
                    template_kwargs.pop("enable_thinking", None)
                    return self.tokenizer.apply_chat_template(messages_dicts, **template_kwargs)

            def _to_chatml_format(self, message: BaseMessage) -> dict:
                if isinstance(message, SystemMessage):
                    role = "system"
                elif isinstance(message, AIMessage):
                    role = "assistant"
                elif isinstance(message, HumanMessage):
                    role = "user"
                else:
                    raise ValueError(f"Unknown message type: {type(message)}")
                return {"role": role, "content": message.content}

        tokenizer = AutoTokenizer.from_pretrained(model)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto",
        )
        text_gen = pipeline(
            "text-generation",
            model=hf_model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            return_full_text=False,  # return only the new answer, not the prompt
        )
        hf_llm = HuggingFacePipeline(pipeline=text_gen)
        return ChatHuggingFaceNoThink(llm=hf_llm, tokenizer=tokenizer)

    if backend == "lmstudio":
        from langchain_openai import ChatOpenAI

        # chat_template_kwargs works on vLLM / transformers serve; LM Studio may ignore it.
        # A larger max_tokens budget still leaves room for the final answer after thinking.
        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key="local",
            temperature=0.1,
            max_tokens=max_new_tokens,
            streaming=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    raise ValueError(f"Unsupported LLM_BACKEND: {backend} (use huggingface, ollama, or lmstudio)")


def vector_count(vectorstore: Chroma) -> int:
    return vectorstore._collection.count()


# -----------------------------
# Indexing
# -----------------------------
def run_index() -> None:
    print("Starting indexing...")
    print(f"DOC_PATH: {DOC_PATH}")
    print(f"PERSIST_DIR: {PERSIST_DIR}")
    print(f"COLLECTION_NAME: {COLLECTION_NAME}")
    print(f"EMBED_MODEL: {EMBED_MODEL}")

    documents = load_documents(DOC_PATH)
    print(f"Loaded docs: {len(documents)}")
    if not documents:
        print("No documents loaded. Check that the file is not empty.")
        sys.exit(1)

    chunks = split_documents(documents)
    print(f"Chunks: {len(chunks)}")
    if not chunks:
        print("No chunks produced. Check splitter settings.")
        sys.exit(1)

    embeddings = build_embeddings()
    vs = build_vectorstore(embeddings)

    # Reset the collection to avoid duplicates across repeated indexing.
    # If you prefer append-only behavior, remove the next 2 lines.
    vs.delete_collection()
    vs = build_vectorstore(embeddings)

    ids = [f"{c.metadata['source']}::{c.metadata['chunk_id']}" for c in chunks]
    vs.add_documents(chunks, ids=ids)

    count = vector_count(vs)
    print(f"Indexed vectors: {count}")
    if count == 0:
        print("Indexing failed: vector count is 0.")
        sys.exit(1)

    print("Indexing completed successfully.")


# -----------------------------
# Tool schema (LangChain tool)
# -----------------------------
class RuleSearchInput(BaseModel):
    """Input schema for the university rules search tool."""

    query: str = Field(
        description="A question or keywords about the university academic regulations."
    )


def build_retrieval_tool(retriever: Optional[VectorStoreRetriever]) -> BaseTool:
    """
    Wrap the retriever as a LangChain tool with an explicit Pydantic schema.
    The graph calls it directly, and it can also be bound to a tool-calling LLM.
    """

    @tool("search_university_rules", args_schema=RuleSearchInput)
    def search_university_rules(query: str) -> List[Document]:
        """Search the university academic regulations and return the most relevant excerpts."""
        if retriever is None:
            return []
        return retriever.invoke(query)

    return search_university_rules


# -----------------------------
# Prompt + helpers
# -----------------------------
SYSTEM_PROMPT = """You are a helpful assistant for university students.
Answer the student's question using ONLY the information in the context.

Rules:
1. Write the answer in your own words as one short, complete paragraph (2-4 sentences).
2. Combine the relevant information from all context excerpts into a single answer.
3. Do NOT copy the excerpts, their headers, or their labels.
4. Mention rule numbers (for example "Rule 1.2") when they support your answer.
5. Do not invent policies, numbers, fees, or deadlines.
6. Do not write a sources list; it is added automatically.
7. If the context does not contain the answer, reply with exactly:
{no_evidence}"""

HUMAN_PROMPT = """Context:
{context}

Question: {question}"""


def build_prompt() -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    return prompt.partial(no_evidence=NO_EVIDENCE_MESSAGE)


def format_docs(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        section = d.metadata.get("section") or d.metadata.get("title") or "General"
        chunk_id = d.metadata.get("chunk_id", "unknown_chunk")
        blocks.append(f"[Excerpt {chunk_id} - {section}]\n{d.page_content}")
    return "\n\n".join(blocks)


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


def clean_answer(text: str) -> str:
    # Drop Qwen3 / Qwen3.5 thinking traces if a backend left them in the answer text.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Remove any sources list the model may have written on its own
    text = re.split(r"\n\s*sources?\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text.strip()


# -----------------------------
# LangGraph: state + nodes + graph
# -----------------------------
class RAGState(TypedDict, total=False):
    """Shared state that flows through the graph for ONE question."""

    question: str
    docs: List[Document]
    answer: str
    final_answer: str


def build_rag_graph(
    retriever: Optional[VectorStoreRetriever],
    llm: Optional[BaseChatModel],
):
    """
    Build and compile the RAG workflow:

      START -> retrieve -> (docs? generate : refuse) -> finalize -> END
    """
    search_tool = build_retrieval_tool(retriever)
    answer_chain = (build_prompt() | llm | StrOutputParser()) if llm is not None else None

    # ---- Nodes (each returns a partial state update) ----
    def retrieve(state: RAGState) -> RAGState:
        docs = search_tool.invoke({"query": state["question"]})
        return {"docs": docs}

    def generate(state: RAGState) -> RAGState:
        answer = answer_chain.invoke(
            {"question": state["question"], "context": format_docs(state["docs"])}
        )
        return {"answer": answer}

    def refuse(state: RAGState) -> RAGState:
        return {"answer": NO_EVIDENCE_MESSAGE}

    def finalize(state: RAGState) -> RAGState:
        docs = state.get("docs", [])
        answer = clean_answer(state.get("answer", ""))
        if not docs or not answer or NO_EVIDENCE_MESSAGE.lower() in answer.lower():
            return {"final_answer": NO_EVIDENCE_MESSAGE}
        return {"final_answer": f"{answer}\n\nSources: {', '.join(unique_sources(docs))}"}

    # ---- Router (conditional edge) ----
    def route_after_retrieve(state: RAGState) -> str:
        return "generate" if state.get("docs") else "refuse"

    # ---- Graph wiring ----
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("refuse", refuse)
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"generate": "generate", "refuse": "refuse"},
    )
    workflow.add_edge("generate", "finalize")
    workflow.add_edge("refuse", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


def ask(graph, question: str) -> str:
    """Run the graph for one question. A fresh state is used every time (stateless)."""
    if not SHOW_STEPS:
        result = graph.invoke({"question": question})
        return result["final_answer"]

    final_answer = NO_EVIDENCE_MESSAGE
    for update in graph.stream({"question": question}, stream_mode="updates"):
        for node_name, node_output in update.items():
            print(f"  [graph] node finished: {node_name}")
            if node_output and "final_answer" in node_output:
                final_answer = node_output["final_answer"]
    return final_answer


# -----------------------------
# Graph visualization
# -----------------------------
def run_graph() -> None:
    """Print and save the graph diagram. No models or database are needed."""
    graph = build_rag_graph(retriever=None, llm=None)
    mermaid = graph.get_graph().draw_mermaid()
    GRAPH_FILE.write_text(mermaid, encoding="utf-8")
    print(mermaid)
    print(f"\nMermaid diagram saved to: {GRAPH_FILE}")
    print("Paste it into https://mermaid.live to view it.")


# -----------------------------
# Chat
# -----------------------------
def run_chat() -> None:
    print("Starting chat mode...")
    print(f"PERSIST_DIR: {PERSIST_DIR}")
    print(f"COLLECTION_NAME: {COLLECTION_NAME}")
    print(f"EMBED_MODEL: {EMBED_MODEL}")
    print(f"LLM_BACKEND: {LLM_BACKEND}")
    print(f"LLM_MODEL: {LLM_MODEL}")

    embeddings = build_embeddings()
    vs = build_vectorstore(embeddings)

    count = vector_count(vs)
    print(f"Existing vectors: {count}")
    if count == 0:
        print("Error: Vector database is empty. Run:")
        print("  python3 rag_agent_local.py index")
        sys.exit(1)

    # Build and compile the graph ONCE; each question runs with a fresh state
    graph = build_rag_graph(build_retriever(vs), build_llm())

    print("\nChat is ready. Type 'exit' to quit.\n")
    try:
        while True:
            question = input("You: ").strip()
            if question.lower() in {"exit", "quit"}:
                print("Bye.")
                break
            if not question:
                continue

            try:
                answer = ask(graph, question)
                print(f"\nAssistant: {answer}\n")
            except Exception as ex:
                print(f"\nAssistant error: {ex}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Bye.")


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 rag_agent_local.py index")
        print("  python3 rag_agent_local.py chat")
        print("  python3 rag_agent_local.py graph")
        print("  python3 rag_agent_local.py web")
        sys.exit(1)

    command = sys.argv[1].strip().lower()

    try:
        if command == "index":
            run_index()
        elif command == "chat":
            run_chat()
        elif command == "graph":
            run_graph()
        elif command == "web":
            import uvicorn
            uvicorn.run("web_api:app", host="127.0.0.1", port=8011)
        else:
            print(f"Unknown command: {command}")
            print("Use 'index', 'chat', 'graph', or 'web'.")
            sys.exit(1)
    except FileNotFoundError as ex:
        print(f"File error: {ex}")
        sys.exit(1)
    except Exception as ex:
        print(f"Fatal error: {ex}")
        sys.exit(1)


if __name__ == "__main__":
    main()
