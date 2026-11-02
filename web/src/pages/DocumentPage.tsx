import {
  BookOpen,
  ChevronRight,
  Database,
  Search,
  SlidersHorizontal,
} from "lucide-react";
import { useMemo } from "react";
import Plot from "react-plotly.js";
import ReactMarkdown from "react-markdown";
import { plotConfig, plotLayout } from "../plot";
import type { Chunk } from "../types";

type Props = {
  document: { name: string; text: string } | null;
  chunks: Chunk[];
  size: number;
  overlap: number;
  selected: number | null;
  filter: string;
  busy: boolean;
  indexProgress: string;
  onSizeChange: (value: number) => void;
  onOverlapChange: (value: number) => void;
  onFilterChange: (value: string) => void;
  onSelect: (id: number) => void;
  onIndex: () => void;
};

export function DocumentPage({
  document,
  chunks,
  size,
  overlap,
  selected,
  filter,
  busy,
  indexProgress,
  onSizeChange,
  onOverlapChange,
  onFilterChange,
  onSelect,
  onIndex,
}: Props) {
  const documentSections = useMemo(
    () => document?.text.match(/^## .+$/gm)?.map((s) => s.slice(3)) || [],
    [document],
  );
  const shownChunks = chunks.filter((c) =>
    (c.text + " " + JSON.stringify(c.metadata))
      .toLowerCase()
      .includes(filter.toLowerCase()),
  );
  const selectedChunk = chunks.find((c) => c.metadata.chunk_id === selected);

  return (
    <div className="document-layout">
      <div className="document-left">
        <section className="card compact">
          <div className="section-kicker">SOURCE DOCUMENT</div>
          <div className="mini-title">
            {document?.name || "Loading…"} <BookOpen size={17} />
          </div>
          <p className="muted">
            Fictional university regulations · educational use only
          </p>
          <div className="section-nav">
            {documentSections.map((section, i) => (
              <button
                key={section}
                onClick={() =>
                  window.document
                    .getElementById(`section-${i}`)
                    ?.scrollIntoView({
                      behavior: "smooth",
                      block: "start",
                    })
                }
              >
                {section}
              </button>
            ))}
          </div>
          <div className="markdown-source">
            <ReactMarkdown
              components={{
                h2: ({ children }) => (
                  <h2
                    id={`section-${documentSections.indexOf(String(children))}`}
                  >
                    {children}
                  </h2>
                ),
              }}
            >
              {document?.text || ""}
            </ReactMarkdown>
          </div>
        </section>
      </div>
      <div className="document-right">
        <section className="card compact">
          <div className="section-kicker">CHUNK CONFIGURATION</div>
          <div className="mini-title">
            Split & explore <SlidersHorizontal size={16} />
          </div>
          <div className="slider-row">
            <label>
              Chunk size <strong>{size}</strong>
            </label>
            <input
              type="range"
              min="100"
              max="1200"
              step="50"
              value={size}
              onChange={(e) => onSizeChange(+e.target.value)}
            />
          </div>
          <div className="slider-row">
            <label>
              Overlap <strong>{overlap}</strong>
            </label>
            <input
              type="range"
              min="0"
              max="500"
              step="25"
              value={overlap}
              onChange={(e) => onOverlapChange(+e.target.value)}
            />
          </div>
          {overlap >= size && (
            <p className="error">Overlap must be smaller than chunk size.</p>
          )}
          <p className="caption">
            Preview updates immediately. Build the index to apply these
            settings.
          </p>
          <button
            className="secondary full"
            disabled={busy || overlap >= size}
            onClick={onIndex}
          >
            <Database size={14} /> Rebuild index
          </button>
          {indexProgress && <p className="caption">{indexProgress}</p>}
        </section>
        <section className="card compact">
          <div className="section-kicker">
            DOCUMENT CHUNKS · {shownChunks.length}
          </div>
          <div className="search-field">
            <Search size={15} />
            <input
              value={filter}
              onChange={(e) => onFilterChange(e.target.value)}
              placeholder="Search chunks or sections"
            />
          </div>
          <div className="chunk-list">
            {shownChunks.map((c) => (
              <button
                key={c.metadata.chunk_id}
                className={
                  "chunk-item " +
                  (selected === c.metadata.chunk_id ? "selected" : "")
                }
                onClick={() => onSelect(c.metadata.chunk_id)}
              >
                <span className="chunk-id">
                  {String(c.metadata.chunk_id).padStart(2, "0")}
                </span>
                <span>
                  <strong>{c.metadata.section || "General"}</strong>
                  <small>{c.text.length} characters</small>
                </span>
                <ChevronRight size={14} />
              </button>
            ))}
          </div>
        </section>
        {selectedChunk && (
          <section className="card compact">
            <div className="section-kicker">
              CHUNK {selectedChunk.metadata.chunk_id}
            </div>
            <div className="mini-title">
              {selectedChunk.metadata.section || "General"}
            </div>
            <pre className="chunk-text">{selectedChunk.text}</pre>
            <div className="adjacent">
              <button
                disabled={selected === 0}
                onClick={() => onSelect((selected || 0) - 1)}
              >
                ← Previous
              </button>
              <button
                disabled={selected === chunks.length - 1}
                onClick={() => onSelect((selected || 0) + 1)}
              >
                Next →
              </button>
            </div>
            {selected !== null && chunks[selected + 1] && (
              <>
                <h4>Compare with next chunk</h4>
                <pre className="chunk-text">{chunks[selected + 1].text}</pre>
              </>
            )}
          </section>
        )}
        <section className="card compact">
          <div className="section-kicker">CHUNK LENGTHS</div>
          <Plot
            data={[
              {
                type: "bar",
                x: chunks.map((c) => c.metadata.chunk_id),
                y: chunks.map((c) => c.text.length),
                marker: { color: "#9c8ce8" },
                hovertemplate: "Chunk %{x}<br>%{y} characters<extra></extra>",
              },
            ]}
            layout={{
              ...plotLayout,
              height: 180,
              xaxis: { ...plotLayout.xaxis, title: "Chunk ID" },
              yaxis: { ...plotLayout.yaxis, title: "Characters" },
            }}
            config={plotConfig}
            style={{ width: "100%" }}
          />
        </section>
      </div>
    </div>
  );
}
