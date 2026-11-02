import { Activity, ChevronRight, FileText, Search } from "lucide-react";
import Plot from "react-plotly.js";
import { plotConfig, plotLayout } from "../plot";
import type { Index, RetrievalResult, Status } from "../types";

type Props = {
  question: string;
  topK: number;
  threshold: number;
  busy: boolean;
  status: Status | null;
  retrieval: RetrievalResult | null;
  onQuestionChange: (value: string) => void;
  onTopKChange: (value: number) => void;
  onThresholdChange: (value: number) => void;
  onSearch: () => void;
  onViewChunk: (id: number, sourceIndex?: Index) => void;
};

export function RetrievalPage({
  question,
  topK,
  threshold,
  busy,
  status,
  retrieval,
  onQuestionChange,
  onTopKChange,
  onThresholdChange,
  onSearch,
  onViewChunk,
}: Props) {
  return (
    <div className="two-col">
      <section className="card compact">
        <div className="section-kicker">QUERY SETTINGS</div>
        <div className="mini-title">
          Test retrieval <Search size={16} />
        </div>
        <p className="muted">
          Search the vector index without generating an answer.
        </p>
        <label className="field-label">QUESTION</label>
        <textarea
          className="input-box"
          value={question}
          onChange={(e) => onQuestionChange(e.target.value)}
          placeholder="Enter a question about the rules…"
          rows={3}
        />
        <div className="slider-row">
          <label>
            Top K <strong>{topK}</strong>
          </label>
          <input
            type="range"
            min="1"
            max="15"
            value={topK}
            onChange={(e) => onTopKChange(+e.target.value)}
          />
        </div>
        <div className="slider-row">
          <label>
            Score threshold <strong>{threshold.toFixed(2)}</strong>
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={threshold}
            onChange={(e) => onThresholdChange(+e.target.value)}
          />
        </div>
        <button
          className="primary full"
          onClick={onSearch}
          disabled={busy || !status?.indexed || !question.trim()}
        >
          <Search size={15} /> Run retrieval
        </button>
        <p className="caption">
          Scores indicate retrieval relevance, not answer confidence.
        </p>
      </section>
      <section className="card compact">
        <div className="section-kicker">SEARCH RESULTS</div>
        <div className="mini-title">
          Relevance ranking <Activity size={17} />
        </div>
        {retrieval ? (
          <>
            <p className="muted">
              Results from the saved query: “{retrieval.settings.question}”
            </p>
            <Plot
              data={[
                {
                  type: "bar",
                  orientation: "h",
                  x: retrieval.hits.map((h) => h.score),
                  y: retrieval.hits.map(
                    (h) => `Chunk ${h.document.metadata.chunk_id}`,
                  ),
                  marker: {
                    color: retrieval.hits.map((h) =>
                      h.accepted ? "#7966d7" : "#d8dce5",
                    ),
                  },
                  hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
                },
              ]}
              layout={{
                ...plotLayout,
                height: Math.max(200, retrieval.hits.length * 38),
                xaxis: {
                  ...plotLayout.xaxis,
                  title: "Relevance score",
                },
                yaxis: { ...plotLayout.yaxis, autorange: "reversed" },
              }}
              config={plotConfig}
              style={{ width: "100%" }}
            />
            {retrieval.hits.map((h) => (
              <button
                className="source-row"
                key={h.document.metadata.chunk_id}
                onClick={() =>
                  onViewChunk(h.document.metadata.chunk_id, retrieval.index)
                }
              >
                <span
                  className={"source-icon " + (!h.accepted ? "dim" : "")}
                >
                  <FileText size={15} />
                </span>
                <span>
                  <strong>
                    {h.document.metadata.section || "General"} · Chunk{" "}
                    {h.document.metadata.chunk_id}
                  </strong>
                  <small>
                    {h.score.toFixed(3)} ·{" "}
                    {h.accepted ? "Included in context" : "Below threshold"}
                  </small>
                </span>
                <ChevronRight size={14} />
              </button>
            ))}
            <h4>Context for model</h4>
            <pre>
              {retrieval.context || "No chunks met the threshold."}
            </pre>
          </>
        ) : (
          <div className="large-empty">
            <Search size={24} />
            <strong>No retrieval yet</strong>
            <span>Enter a question to inspect its nearest chunks.</span>
          </div>
        )}
      </section>
    </div>
  );
}
