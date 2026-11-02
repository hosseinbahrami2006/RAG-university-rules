export type Meta = {
  chunk_id: number;
  section?: string;
  title?: string;
  source: string;
};

export type Chunk = { text: string; metadata: Meta };

export type Hit = { document: Chunk; score: number; accepted: boolean };

export type Index = {
  id: string;
  count: number;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
};

export type Status = {
  indexed: boolean;
  index: Index | null;
  busy: boolean;
  embedding_model: string;
  llm_model: string;
};

export type Settings = { question: string; top_k: number; score_threshold: number };

export type Trace = {
  input?: unknown;
  output?: unknown;
  duration_ms?: number;
  status: string;
};

export type Run = {
  id: string;
  question: string;
  settings: Settings;
  index: Index;
  answer: string;
  hits: Hit[];
  context: string;
  traces: Record<string, Trace>;
  duration_ms?: number;
  error?: string;
};

export type Projection = {
  points: Array<{ x: number; y: number; text: string; metadata: Meta }>;
  similarity: number[][];
  variance: number[];
  index: Index;
};

export type RetrievalResult = {
  hits: Hit[];
  context: string;
  settings: Settings;
  index: Index;
};

export type PageId =
  | "ask"
  | "workflow"
  | "document"
  | "retrieval"
  | "embeddings";
