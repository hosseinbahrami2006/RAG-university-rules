import { useEffect, useState } from "react";
import "@xyflow/react/dist/style.css";
import { api, stream } from "./api";
import { Footer } from "./components/Footer";
import { Notice } from "./components/Notice";
import { PageHeading } from "./components/PageHeading";
import { SetupBanner } from "./components/SetupBanner";
import { Sidebar } from "./components/Sidebar";
import { Topbar } from "./components/Topbar";
import { initialNodePositions, initialTrace } from "./constants";
import { AskPage } from "./pages/AskPage";
import { DocumentPage } from "./pages/DocumentPage";
import { EmbeddingsPage } from "./pages/EmbeddingsPage";
import { RetrievalPage } from "./pages/RetrievalPage";
import { WorkflowPage } from "./pages/WorkflowPage";
import type {
  Chunk,
  Index,
  PageId,
  Projection,
  RetrievalResult,
  Run,
  Status,
} from "./types";

export function App() {
  const [page, setPage] = useState<PageId>("ask");
  const [status, setStatus] = useState<Status | null>(null);
  const [document, setDocument] = useState<{ name: string; text: string } | null>(
    null,
  );
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [size, setSize] = useState(600);
  const [overlap, setOverlap] = useState(100);
  const [selected, setSelected] = useState<number | null>(null);
  const [filter, setFilter] = useState("");
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(4);
  const [threshold, setThreshold] = useState(0.25);
  const [runs, setRuns] = useState<Run[]>([]);
  const [currentId, setCurrentId] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [indexProgress, setIndexProgress] = useState("");
  const [retrieval, setRetrieval] = useState<RetrievalResult | null>(null);
  const [projection, setProjection] = useState<Projection | null>(null);
  const [queryPoint, setQueryPoint] = useState<{ x: number; y: number } | null>(
    null,
  );
  const [node, setNode] = useState("retrieve");
  const [nodePositions, setNodePositions] = useState(initialNodePositions);

  const current = runs.find((r) => r.id === currentId);
  const settings = { question, top_k: topK, score_threshold: threshold };

  const refresh = () =>
    api<Status>("/status")
      .then(setStatus)
      .catch((e) => setNotice(e.message));

  useEffect(() => {
    refresh();
    api<{ name: string; text: string }>("/document")
      .then(setDocument)
      .catch((e) => setNotice(e.message));
  }, []);

  useEffect(() => {
    if (overlap >= size) return;
    const t = setTimeout(
      () =>
        api<{ chunks: Chunk[] }>("/chunks/preview", {
          chunk_size: size,
          chunk_overlap: overlap,
        })
          .then((r) => setChunks(r.chunks))
          .catch((e) => setNotice(e.message)),
      250,
    );
    return () => clearTimeout(t);
  }, [size, overlap]);

  const updateRun = (id: string, fn: (r: Run) => Run) =>
    setRuns((rs) => rs.map((r) => (r.id === id ? fn(r) : r)));

  async function index() {
    setBusy(true);
    setNotice("");
    setIndexProgress("Starting…");
    let complete = false;
    try {
      await stream(
        "/index",
        { chunk_size: size, chunk_overlap: overlap },
        (e) => {
          if (e.type === "progress") setIndexProgress(e.message);
          if (e.type === "error") throw Error(e.message);
          if (e.type === "complete") {
            complete = true;
            setIndexProgress("Index ready");
            setProjection(null);
            setRetrieval(null);
          }
        },
      );
      if (!complete) throw Error("Indexing ended without a completion event.");
      await refresh();
    } catch (e) {
      setNotice(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function ask() {
    if (!question.trim() || busy) return;
    setBusy(true);
    setNotice("");
    setQueryPoint(null);
    let id = "";
    try {
      await stream("/ask", settings, (e) => {
        if (e.type === "start") {
          id = e.run_id;
          setCurrentId(id);
          setRuns((rs) => [
            {
              id,
              question: e.settings.question,
              settings: e.settings,
              index: e.index,
              answer: "",
              hits: [],
              context: "",
              traces: initialTrace(),
            },
            ...rs,
          ]);
        } else if (e.type === "token")
          updateRun(id, (r) => ({ ...r, answer: r.answer + e.text }));
        else if (e.type === "node_start")
          updateRun(id, (r) => ({
            ...r,
            traces: {
              ...r.traces,
              [e.node]: {
                ...r.traces[e.node],
                status: "running",
                input: e.input,
              },
            },
          }));
        else if (e.type === "node_complete")
          updateRun(id, (r) => ({
            ...r,
            hits: e.hits ?? r.hits,
            context: e.context ?? r.context,
            traces: {
              ...r.traces,
              [e.node]: {
                ...r.traces[e.node],
                status: "complete",
                output: e.output,
                duration_ms: e.duration_ms,
              },
            },
          }));
        else if (e.type === "complete")
          updateRun(id, (r) => ({
            ...r,
            answer: e.answer,
            duration_ms: e.duration_ms,
          }));
        else if (e.type === "error") {
          updateRun(id, (r) => ({ ...r, error: e.message }));
          setNotice(e.message);
        }
      });
    } catch (e) {
      setNotice(String(e));
    } finally {
      setBusy(false);
      refresh();
    }
  }

  async function search() {
    if (!question.trim() || busy) return;
    setBusy(true);
    setNotice("");
    try {
      const result = await api<RetrievalResult>("/retrieve", settings);
      setRetrieval(result);
      setQueryPoint(null);
    } catch (e) {
      setNotice(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function loadProjection() {
    setBusy(true);
    setNotice("");
    try {
      setProjection(await api<Projection>("/embeddings"));
    } catch (e) {
      setNotice(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function showQuery() {
    const q =
      current && current.index.id === projection?.index.id
        ? current.settings
        : retrieval && retrieval.index.id === projection?.index.id
          ? retrieval.settings
          : settings;
    if (!q.question.trim()) return;
    setBusy(true);
    try {
      setQueryPoint(
        await api<{ x: number; y: number }>("/embeddings/query", q),
      );
    } catch (e) {
      setNotice(String(e));
    } finally {
      setBusy(false);
    }
  }

  function viewChunk(
    id: number,
    sourceIndex: Index | undefined = current?.index || retrieval?.index,
  ) {
    if (
      sourceIndex &&
      (size !== sourceIndex.chunk_size || overlap !== sourceIndex.chunk_overlap)
    ) {
      setSize(sourceIndex.chunk_size);
      setOverlap(sourceIndex.chunk_overlap);
    }
    setSelected(id);
    setPage("document");
  }

  return (
    <div className="app-shell">
      <Sidebar page={page} onNavigate={setPage} />
      <main className="main">
        <Topbar page={page} status={status} />
        <div className="content">
          <PageHeading
            page={page}
            onViewSource={() => setPage("document")}
          />
          {notice && (
            <Notice message={notice} onDismiss={() => setNotice("")} />
          )}
          {!status?.indexed && (
            <SetupBanner
              busy={busy}
              disabled={overlap >= size}
              indexProgress={indexProgress}
              onIndex={index}
            />
          )}
          {page === "ask" && (
            <AskPage
              runs={runs}
              current={current}
              currentId={currentId}
              question={question}
              busy={busy}
              status={status}
              onQuestionChange={setQuestion}
              onSelectRun={setCurrentId}
              onAsk={ask}
              onViewWorkflow={() => setPage("workflow")}
              onViewChunk={viewChunk}
            />
          )}
          {page === "workflow" && (
            <WorkflowPage
              current={current}
              selectedNode={node}
              nodePositions={nodePositions}
              onSelectNode={setNode}
              onNodePositionsChange={setNodePositions}
            />
          )}
          {page === "document" && (
            <DocumentPage
              document={document}
              chunks={chunks}
              size={size}
              overlap={overlap}
              selected={selected}
              filter={filter}
              busy={busy}
              indexProgress={indexProgress}
              onSizeChange={setSize}
              onOverlapChange={setOverlap}
              onFilterChange={setFilter}
              onSelect={setSelected}
              onIndex={index}
            />
          )}
          {page === "retrieval" && (
            <RetrievalPage
              question={question}
              topK={topK}
              threshold={threshold}
              busy={busy}
              status={status}
              retrieval={retrieval}
              onQuestionChange={setQuestion}
              onTopKChange={setTopK}
              onThresholdChange={setThreshold}
              onSearch={search}
              onViewChunk={viewChunk}
            />
          )}
          {page === "embeddings" && (
            <EmbeddingsPage
              busy={busy}
              indexed={!!status?.indexed}
              projection={projection}
              queryPoint={queryPoint}
              question={question}
              current={current}
              retrieval={retrieval}
              onLoadProjection={loadProjection}
              onShowQuery={showQuery}
              onViewChunk={viewChunk}
            />
          )}
          <Footer />
        </div>
      </main>
    </div>
  );
}
