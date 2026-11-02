import {
  ReactFlow,
  Background,
  Controls,
  type Edge,
  type Node,
  type NodeChange,
} from "@xyflow/react";
import { Braces, GitBranch } from "lucide-react";
import { pipelineNodes } from "../constants";
import type { Run } from "../types";

type Props = {
  current: Run | undefined;
  selectedNode: string;
  nodePositions: Record<string, { x: number; y: number }>;
  onSelectNode: (id: string) => void;
  onNodePositionsChange: (
    updater: (
      prev: Record<string, { x: number; y: number }>,
    ) => Record<string, { x: number; y: number }>,
  ) => void;
};

export function WorkflowPage({
  current,
  selectedNode,
  nodePositions,
  onSelectNode,
  onNodePositionsChange,
}: Props) {
  const nodes: Node[] = pipelineNodes.map((n) => ({
    id: n,
    position: nodePositions[n],
    data: {
      label: (
        <div className="flow-label">
          <span
            className={
              "flow-dot " + (current?.traces[n]?.status || "waiting")
            }
          ></span>
          <div>
            <strong>{n}</strong>
            <small>
              {current?.traces[n]?.duration_ms !== undefined
                ? `${current.traces[n].duration_ms} ms`
                : current?.traces[n]?.status || "waiting"}
            </small>
          </div>
        </div>
      ),
    },
    className: `flow-node ${selectedNode === n ? "selected" : ""} ${current?.traces[n]?.status || ""}`,
  }));

  const edges: Edge[] = [
    {
      id: "rg",
      source: "retrieve",
      target: "generate",
      label: "evidence found",
    },
    { id: "rr", source: "retrieve", target: "refuse", label: "no evidence" },
    { id: "gf", source: "generate", target: "finalize" },
    { id: "rf", source: "refuse", target: "finalize" },
  ].map((e) => ({
    ...e,
    animated: current?.traces[e.source]?.status === "running",
    style: { stroke: "#b9c6d5", strokeWidth: 1.5 },
    labelStyle: { fontSize: 10, fill: "#7a899b" },
  }));

  const trace = current?.traces[selectedNode];

  return (
    <div className="two-col workflow-grid">
      <section className="card">
        <div className="card-header">
          <div>
            <div className="card-title">
              <span className="icon-square purple">
                <GitBranch size={16} />
              </span>{" "}
              Pipeline execution
            </div>
            <p>Drag to pan, scroll to zoom, click a node to inspect.</p>
          </div>
          <span className="tag">LANGGRAPH</span>
        </div>
        <div className="flow-canvas">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            fitView
            onNodeClick={(_, n) => onSelectNode(n.id)}
            onNodesChange={(changes: NodeChange[]) =>
              onNodePositionsChange((prev) => {
                const next = { ...prev };
                for (const change of changes)
                  if (change.type === "position" && change.position)
                    next[change.id] = change.position;
                return next;
              })
            }
            nodesDraggable={true}
          >
            <Background gap={18} size={1} />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>
        <div className="legend">
          <span>
            <i className="legend-dot green"></i> Complete
          </span>
          <span>
            <i className="legend-dot purple-bg"></i> Running
          </span>
          <span>
            <i className="legend-dot gray"></i> Not executed
          </span>
        </div>
      </section>
      <section className="card compact">
        <div className="section-kicker">NODE INSPECTOR</div>
        <div className="mini-title">
          {selectedNode} <Braces size={17} />
        </div>
        <p className="muted">
          {current
            ? `Selected run · ${current.question}`
            : "Run a question to inspect its execution."}
        </p>
        {trace && (
          <>
            <div className="overview-row">
              <span>STATUS</span>
              <strong>{trace.status}</strong>
            </div>
            <div className="overview-row">
              <span>DURATION</span>
              <strong>{trace.duration_ms ?? "—"} ms</strong>
            </div>
            <h4>Input</h4>
            <pre>{JSON.stringify(trace.input ?? {}, null, 2)}</pre>
            <h4>Output</h4>
            <pre>{JSON.stringify(trace.output ?? {}, null, 2)}</pre>
          </>
        )}
      </section>
    </div>
  );
}
