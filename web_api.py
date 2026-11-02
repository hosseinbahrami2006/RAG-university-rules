"""Local browser API for the university RAG workflow."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field, model_validator

import rag_agent_local as rag

app = FastAPI(title='University RAG Lab')
META_PATH = Path(rag.PERSIST_DIR) / 'web-index.json'
MODEL_NAME = os.getenv('LLM_MODEL', 'qwen3.5-4b')
BASE_URL = os.getenv('LMSTUDIO_BASE_URL', 'http://127.0.0.1:1234/v1')
operation_lock = threading.Lock()
_embeddings = None


class ChunkSettings(BaseModel):
    chunk_size: int = Field(default=600, ge=100, le=2000)
    chunk_overlap: int = Field(default=100, ge=0, le=1000)

    @model_validator(mode='after')
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('Overlap must be smaller than chunk size')
        return self


class SearchSettings(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=30)
    score_threshold: float = Field(default=0.25, ge=0, le=1)

    @model_validator(mode='after')
    def trim_question(self):
        self.question = self.question.strip()
        if not self.question:
            raise ValueError('Enter a question')
        return self


def document_hash():
    return hashlib.sha256(Path(rag.DOC_PATH).read_bytes()).hexdigest()


def index_meta():
    try:
        return json.loads(META_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def index_valid(meta):
    return bool(meta and meta.get('document_hash') == document_hash() and meta.get('embedding_model') == rag.EMBED_MODEL)


def embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = rag.build_embeddings()
    return _embeddings


def store(meta):
    if not index_valid(meta):
        raise HTTPException(409, 'Index missing or stale. Build the index first.')
    return rag.build_vectorstore(embeddings(), collection_name=meta['collection'])


def doc_json(doc):
    return {'text': doc.page_content, 'metadata': doc.metadata}


def sse(kind, **payload):
    return 'data: ' + json.dumps({'type': kind, **payload}, ensure_ascii=False, default=str) + '\n\n'


def serialize(value):
    if isinstance(value, Document):
        return doc_json(value)
    if isinstance(value, dict):
        return {k: serialize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialize(v) for v in value]
    return value


class ScoredRetriever(BaseRetriever):
    vectorstore: Any
    top_k: int
    threshold: float
    hits: list = Field(default_factory=list)

    def _get_relevant_documents(self, query: str, *, run_manager):
        # This is the exact query used for graph context and visualization.
        pairs = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.top_k)
        self.hits = [{'document': doc_json(doc), 'score': float(score), 'accepted': score >= self.threshold} for doc, score in pairs]
        return [doc for doc, score in pairs if score >= self.threshold]


def scored_search(vs, request):
    retriever = ScoredRetriever(vectorstore=vs, top_k=request.top_k, threshold=request.score_threshold)
    docs = retriever.invoke(request.question)
    return retriever, docs


@app.get('/api/status')
def status():
    meta = index_meta()
    return {'indexed': index_valid(meta), 'index': meta, 'busy': operation_lock.locked(), 'embedding_model': rag.EMBED_MODEL, 'llm_model': MODEL_NAME, 'lmstudio_url': BASE_URL}


@app.get('/api/document')
def document():
    text = Path(rag.DOC_PATH).read_text(encoding='utf-8')
    return {'name': Path(rag.DOC_PATH).name, 'text': text, 'hash': document_hash()}


@app.post('/api/chunks/preview')
def preview(settings: ChunkSettings):
    chunks = rag.split_documents(rag.load_documents(rag.DOC_PATH), settings.chunk_size, settings.chunk_overlap)
    return {'chunks': [doc_json(c) for c in chunks], 'settings': settings.model_dump()}


@app.get('/api/graph')
def graph():
    g = rag.build_rag_graph(None, None)
    return {'mermaid': g.get_graph().draw_mermaid(), 'nodes': ['retrieve', 'generate', 'refuse', 'finalize']}


@app.post('/api/index')
def build_index(settings: ChunkSettings):
    if not operation_lock.acquire(blocking=False):
        raise HTTPException(409, 'Another operation is running.')
    def stream():
        collection = 'web_' + uuid.uuid4().hex
        vs = None
        try:
            yield sse('progress', message='Splitting document')
            source_hash = document_hash()
            docs = rag.split_documents(rag.load_documents(rag.DOC_PATH), settings.chunk_size, settings.chunk_overlap)
            if not docs:
                raise ValueError('The document produced no chunks.')
            yield sse('progress', message=f'Loading embedding model and indexing {len(docs)} chunks')
            vs = rag.build_vectorstore(embeddings(), collection_name=collection)
            vs.add_documents(docs, ids=[f"{d.metadata['source']}::{d.metadata['chunk_id']}" for d in docs])
            count = rag.vector_count(vs)
            if count != len(docs):
                raise ValueError('Index count does not match the number of chunks.')
            if document_hash() != source_hash:
                raise ValueError('Source document changed while indexing. Please rebuild.')
            meta = {'id': uuid.uuid4().hex, 'collection': collection, 'document_hash': source_hash, 'embedding_model': rag.EMBED_MODEL, 'chunk_size': settings.chunk_size, 'chunk_overlap': settings.chunk_overlap, 'count': count}
            META_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = META_PATH.with_suffix('.tmp')
            tmp.write_text(json.dumps(meta), encoding='utf-8')
            tmp.replace(META_PATH)
            yield sse('complete', index=meta)
        except Exception as exc:
            if vs is not None:
                try: vs.delete_collection()
                except Exception: pass
            yield sse('error', message=str(exc))
        finally:
            operation_lock.release()
    return StreamingResponse(stream(), media_type='text/event-stream')


@app.post('/api/retrieve')
def retrieve(request: SearchSettings):
    if not operation_lock.acquire(blocking=False):
        raise HTTPException(409, 'Another operation is running.')
    try:
        meta = index_meta()
        retriever, docs = scored_search(store(meta), request)
        return {'hits': retriever.hits, 'context': rag.format_docs(docs), 'settings': request.model_dump(), 'index': meta}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    finally:
        operation_lock.release()


@app.post('/api/ask')
def ask(request: SearchSettings):
    if not operation_lock.acquire(blocking=False):
        raise HTTPException(409, 'Another operation is running.')
    meta = index_meta()
    if not index_valid(meta):
        operation_lock.release()
        raise HTTPException(409, 'Index missing or stale. Build the index first.')
    run_id = uuid.uuid4().hex
    def stream():
        started = time.monotonic()
        state = {'question': request.question}
        timings = {}
        retriever = None
        try:
            yield sse('start', run_id=run_id, settings=request.model_dump(), index=meta)
            retriever = ScoredRetriever(vectorstore=store(meta), top_k=request.top_k, threshold=request.score_threshold)
            llm = rag.build_llm('lmstudio', MODEL_NAME, BASE_URL)
            graph = rag.build_rag_graph(retriever, llm)
            for part in graph.stream({'question': request.question}, stream_mode=['tasks', 'updates', 'messages'], version='v2'):
                kind, data = part['type'], part['data']
                if kind == 'tasks' and data.get('name') in ('retrieve', 'generate', 'refuse', 'finalize'):
                    name = data['name']
                    if 'result' not in data:
                        timings[name] = time.monotonic()
                        yield sse('node_start', run_id=run_id, node=name, input=serialize(state))
                elif kind == 'updates':
                    for name, update in data.items():
                        if name not in ('retrieve', 'generate', 'refuse', 'finalize') or not isinstance(update, dict):
                            continue
                        state.update(update)
                        extra = {}
                        if name == 'retrieve':
                            extra = {'hits': retriever.hits, 'context': rag.format_docs(update.get('docs', []))}
                        yield sse('node_complete', run_id=run_id, node=name, output=serialize(update), duration_ms=round((time.monotonic()-timings.get(name, started))*1000), **extra)
                elif kind == 'messages':
                    message, metadata = data
                    if metadata.get('langgraph_node') == 'generate' and isinstance(message.content, str) and message.content:
                        yield sse('token', run_id=run_id, text=message.content)
            yield sse('complete', run_id=run_id, answer=state.get('final_answer', rag.NO_EVIDENCE_MESSAGE), duration_ms=round((time.monotonic()-started)*1000))
        except Exception as exc:
            yield sse('error', run_id=run_id, message=f'{type(exc).__name__}: {exc}')
        finally:
            operation_lock.release()
    return StreamingResponse(stream(), media_type='text/event-stream')


@app.get('/api/embeddings')
def projection():
    if not operation_lock.acquire(blocking=False):
        raise HTTPException(409, 'Another operation is running.')
    try:
        meta = index_meta()
        vs = store(meta)
        records = vs._collection.get(include=['embeddings', 'documents', 'metadatas'])
        vectors = np.array(records['embeddings'], dtype=float)
        if vectors.ndim != 2 or len(vectors) < 2:
            raise HTTPException(409, 'At least two indexed chunks are needed for PCA.')
        centered = vectors - vectors.mean(axis=0)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        points = centered @ vt[:2].T
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        similarity = (vectors / np.maximum(norm, 1e-12)) @ (vectors / np.maximum(norm, 1e-12)).T
        return {'points': [{'x': float(p[0]), 'y': float(p[1]) if len(p)>1 else 0, 'metadata': m, 'text': t} for p, m, t in zip(points, records['metadatas'], records['documents'])], 'similarity': similarity.tolist(), 'variance': (singular[:2]**2 / max(float((singular**2).sum()), 1e-12)).tolist(), 'index': meta}
    finally:
        operation_lock.release()


@app.post('/api/embeddings/query')
def query_projection(request: SearchSettings):
    if not operation_lock.acquire(blocking=False):
        raise HTTPException(409, 'Another operation is running.')
    try:
        vs = store(index_meta())
        records = vs._collection.get(include=['embeddings'])
        vectors = np.array(records['embeddings'], dtype=float)
        if vectors.ndim != 2 or len(vectors) < 2:
            raise HTTPException(409, 'At least two indexed chunks are needed for PCA.')
        mean = vectors.mean(axis=0)
        _, _, vt = np.linalg.svd(vectors-mean, full_matrices=False)
        query = np.array(embeddings().embed_query(request.question))
        point = (query-mean) @ vt[:2].T
        return {'x': float(point[0]), 'y': float(point[1]) if len(point)>1 else 0}
    finally:
        operation_lock.release()


DIST = Path(__file__).parent / 'web' / 'dist'
if DIST.exists():
    app.mount('/assets', StaticFiles(directory=DIST / 'assets'), name='assets')
    @app.get('/')
    def home():
        return FileResponse(DIST / 'index.html')
