import {
  BookOpen,
  Boxes,
  GitBranch,
  MessageSquare,
  SlidersHorizontal,
} from "lucide-react";
import type { PageId } from "./types";

export const examples = [
  "What is the late registration deadline and fee?",
  "How many credits can a student on probation take?",
  "What happens if I miss a final exam because of an emergency?",
  "Are AI tools allowed in assignments?",
];

export const menu = [
  ["ask", "Ask & inspect", MessageSquare],
  ["workflow", "Live workflow", GitBranch],
  ["document", "Document & chunks", BookOpen],
  ["retrieval", "Retrieval playground", SlidersHorizontal],
  ["embeddings", "Embedding explorer", Boxes],
] as const satisfies ReadonlyArray<
  readonly [PageId, string, typeof MessageSquare]
>;

export const titles: Record<PageId, [string, string]> = {
  ask: ["Ask & inspect", "Explore answers grounded in university regulations."],
  workflow: [
    "Live workflow",
    "Follow each step of the RAG pipeline as it happens.",
  ],
  document: [
    "Document & chunks",
    "See how the source document becomes searchable context.",
  ],
  retrieval: [
    "Retrieval playground",
    "Tune the search and inspect the evidence before generation.",
  ],
  embeddings: [
    "Embedding explorer",
    "Explore the relationships between indexed document chunks.",
  ],
};

export const initialNodePositions: Record<string, { x: number; y: number }> = {
  retrieve: { x: 270, y: 35 },
  generate: { x: 100, y: 180 },
  refuse: { x: 440, y: 180 },
  finalize: { x: 270, y: 325 },
};

export const pipelineNodes = ["retrieve", "generate", "refuse", "finalize"] as const;

export function initialTrace() {
  return Object.fromEntries(
    pipelineNodes.map((n) => [n, { status: "waiting" }]),
  );
}
