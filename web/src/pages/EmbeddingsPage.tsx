import { Activity, Boxes, Layers3, Search } from "lucide-react";
import { useMemo } from "react";
import Plot from "react-plotly.js";
import { plotConfig, plotLayout } from "../plot";
import type { Hit, Index, Projection, Run, RetrievalResult } from "../types";

type Props = {
  busy: boolean;
  indexed: boolean;
  projection: Projection | null;
  queryPoint: { x: number; y: number } | null;
  question: string;
  current: Run | undefined;
  retrieval: RetrievalResult | null;
  onLoadProjection: () => void;
  onShowQuery: () => void;
  onViewChunk: (id: number, sourceIndex?: Index) => void;
};

export function EmbeddingsPage({
  busy,
  indexed,
  projection,
  queryPoint,
  question,
  current,
  retrieval,
  onLoadProjection,
  onShowQuery,
  onViewChunk,
}: Props) {
  const sections = useMemo(
    () =>
      Array.from(
        new Set(
          projection?.points.map((p) => p.metadata.section || "General") || [],
        ),
      ),
    [projection],
  );

  const activeHits: Hit[] | undefined =
    current?.index.id === projection?.index.id
      ? current?.hits
      : retrieval?.index.id === projection?.index.id
        ? retrieval?.hits
        : undefined;

  return (
    <div className="embedding-stack">
      <section className="card compact">
        <div className="card-header no-pad">
          <div>
            <div className="card-title">
              <span className="icon-square purple">
                <Boxes size={16} />
              </span>{" "}
              Embedding space
            </div>
            <p>Each point represents a real indexed chunk.</p>
          </div>
          <div className="inline-actions">
            <button
              className="secondary"
              onClick={onLoadProjection}
              disabled={busy || !indexed}
            >
              <Activity size={14} />{" "}
              {projection ? "Refresh map" : "Load map"}
            </button>
            <button
              className="secondary"
              onClick={onShowQuery}
              disabled={
                busy ||
                !projection ||
                !(
                  current?.question ||
                  retrieval?.settings.question ||
                  question.trim()
                )
              }
            >
              <Search size={14} /> Show query
            </button>
          </div>
        </div>
        {projection ? (
          <Plot
            data={
              [
                ...sections.map((section, i) => ({
                  type: "scatter" as const,
                  mode: "markers" as const,
                  name: section,
                  x: projection.points
                    .filter(
                      (p) => (p.metadata.section || "General") === section,
                    )
                    .map((p) => p.x),
                  y: projection.points
                    .filter(
                      (p) => (p.metadata.section || "General") === section,
                    )
                    .map((p) => p.y),
                  text: projection.points
                    .filter(
                      (p) => (p.metadata.section || "General") === section,
                    )
                    .map(
                      (p) => `Chunk ${p.metadata.chunk_id}: ${section}`,
                    ),
                  customdata: projection.points
                    .filter(
                      (p) => (p.metadata.section || "General") === section,
                    )
                    .map((p) => p.metadata.chunk_id),
                  marker: {
                    size: projection.points
                      .filter(
                        (p) => (p.metadata.section || "General") === section,
                      )
                      .map((p) =>
                        activeHits?.some(
                          (h) =>
                            h.accepted &&
                            h.document.metadata.chunk_id ===
                              p.metadata.chunk_id,
                        )
                          ? 15
                          : 10,
                      ),
                    color: [
                      "#806dd4",
                      "#55a0b6",
                      "#e1aa69",
                      "#df8299",
                      "#6aab88",
                      "#8f9dca",
                      "#be91bf",
                      "#80bcb4",
                      "#d89d80",
                      "#a9a777",
                      "#b4a2ce",
                    ][i % 11],
                    line: { color: "#fff", width: 1.5 },
                  },
                  hovertemplate: "%{text}<extra></extra>",
                })),
                ...(queryPoint
                  ? [
                      {
                        type: "scatter" as const,
                        mode: "markers" as const,
                        name: "Question",
                        x: [queryPoint.x],
                        y: [queryPoint.y],
                        marker: {
                          size: 16,
                          color: "#211d39",
                          symbol: "star",
                        },
                        hovertemplate: "Question<extra></extra>",
                      },
                    ]
                  : []),
              ] as any
            }
            layout={{
              ...plotLayout,
              height: 480,
              showlegend: true,
              legend: { orientation: "h", y: -0.2 },
              xaxis: {
                ...plotLayout.xaxis,
                title: `PC1 (${((projection.variance[0] || 0) * 100).toFixed(1)}%)`,
              },
              yaxis: {
                ...plotLayout.yaxis,
                title: `PC2 (${((projection.variance[1] || 0) * 100).toFixed(1)}%)`,
              },
            }}
            config={plotConfig}
            style={{ width: "100%" }}
            onClick={(e: any) => {
              const id = e.points[0]?.customdata;
              if (typeof id === "number") onViewChunk(id, projection.index);
            }}
          />
        ) : (
          <div className="large-empty">
            <Boxes size={28} />
            <strong>Explore the embedding space</strong>
            <span>Load the map after building your index.</span>
          </div>
        )}
        <p className="caption">
          PCA is a two-dimensional approximation. Larger points are accepted
          results from the selected run or retrieval.
        </p>
      </section>
      {projection && (
        <section className="card compact">
          <div className="section-kicker">PAIRWISE COSINE SIMILARITY</div>
          <div className="mini-title">
            Chunk similarity matrix <Layers3 size={17} />
          </div>
          <Plot
            data={[
              {
                type: "heatmap",
                z: projection.similarity,
                x: projection.points.map((p) => p.metadata.chunk_id),
                y: projection.points.map((p) => p.metadata.chunk_id),
                colorscale: [
                  [0, "#f6f4fc"],
                  [1, "#6a56c2"],
                ],
                zmin: -1,
                zmax: 1,
                hovertemplate:
                  "Chunk %{x} / %{y}<br>Cosine similarity %{z:.3f}<extra></extra>",
              },
            ]}
            layout={{
              ...plotLayout,
              height: 520,
              xaxis: { ...plotLayout.xaxis, title: "Chunk ID" },
              yaxis: { ...plotLayout.yaxis, title: "Chunk ID" },
            }}
            config={plotConfig}
            style={{ width: "100%" }}
          />
        </section>
      )}
    </div>
  );
}
