import {
  Activity,
  ArrowRight,
  ChevronRight,
  Command,
  FileText,
  Layers3,
  LoaderCircle,
  MessageSquare,
  Send,
  Sparkles,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import { examples } from "../constants";
import type { Hit, Index, Run, Status } from "../types";

type Props = {
  runs: Run[];
  current: Run | undefined;
  currentId: string;
  question: string;
  busy: boolean;
  status: Status | null;
  onQuestionChange: (value: string) => void;
  onSelectRun: (id: string) => void;
  onAsk: () => void;
  onViewWorkflow: () => void;
  onViewChunk: (id: number, sourceIndex?: Index) => void;
};

export function AskPage({
  runs,
  current,
  currentId,
  question,
  busy,
  status,
  onQuestionChange,
  onSelectRun,
  onAsk,
  onViewWorkflow,
  onViewChunk,
}: Props) {
  const acceptedHits = current?.hits.filter((h) => h.accepted) ?? [];

  return (
    <div className="ask-grid">
      <section className="card chat-card">
        <div className="card-header">
          <div>
            <div className="card-title">
              <span className="icon-square purple">
                <MessageSquare size={16} />
              </span>{" "}
              Conversation
            </div>
            <p>Ask a question about the regulations</p>
          </div>
          <span className="tag">LOCAL RAG</span>
        </div>
        <div className="conversation">
          {runs.length === 0 ? (
            <EmptyChat onPickExample={onQuestionChange} />
          ) : (
            <div className="messages">
              {runs
                .slice()
                .reverse()
                .map((r) => (
                  <RunMessage
                    key={r.id}
                    run={r}
                    selected={currentId === r.id}
                    onSelect={() => onSelectRun(r.id)}
                    onViewChunk={onViewChunk}
                  />
                ))}
            </div>
          )}
        </div>
        <div className="composer">
          <textarea
            value={question}
            onChange={(e) => onQuestionChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onAsk();
              }
            }}
            placeholder="Ask about university regulations…"
            rows={2}
          />
          <div className="composer-footer">
            <span>
              <Command size={13} /> Enter to send · Shift + Enter for a new line
            </span>
            <button
              className="primary"
              disabled={busy || !status?.indexed || !question.trim()}
              onClick={onAsk}
            >
              {busy ? (
                <LoaderCircle className="spin" size={15} />
              ) : (
                <Send size={15} />
              )}{" "}
              Ask question
            </button>
          </div>
        </div>
      </section>
      <aside className="right-stack">
        <section className="card compact">
          <div className="section-kicker">CURRENT RUN</div>
          <div className="mini-title">
            Execution overview <Activity size={16} />
          </div>
          <p className="muted">
            {current
              ? "A snapshot of the selected question."
              : "Your execution details will appear here."}
          </p>
          <div className="overview-row">
            <span>STATUS</span>
            <strong>
              <span
                className={
                  "status-dot " + (current?.duration_ms ? "green" : "amber")
                }
              ></span>
              {current?.error
                ? "Error"
                : current?.duration_ms
                  ? "Complete"
                  : current
                    ? "Running"
                    : "Waiting"}
            </strong>
          </div>
          <div className="overview-row">
            <span>RETRIEVED</span>
            <strong>
              {current ? `${acceptedHits.length} chunks` : "—"}
            </strong>
          </div>
          <div className="overview-row">
            <span>DURATION</span>
            <strong>
              {current?.duration_ms
                ? `${(current.duration_ms / 1000).toFixed(1)} s`
                : "—"}
            </strong>
          </div>
          <button className="text-button" onClick={onViewWorkflow}>
            View workflow <ArrowRight size={14} />
          </button>
        </section>
        <section className="card compact">
          <div className="section-kicker">RETRIEVED EVIDENCE</div>
          <div className="mini-title">
            Sources <Layers3 size={16} />
          </div>
          {acceptedHits.length ? (
            acceptedHits.map((h) => (
              <SourceRow
                key={h.document.metadata.chunk_id}
                hit={h}
                onClick={() => onViewChunk(h.document.metadata.chunk_id)}
              />
            ))
          ) : (
            <div className="source-empty">
              <Layers3 size={22} />
              <span>Relevant sources will show up here.</span>
            </div>
          )}
          <p className="caption">
            Sources are retrieved context, not verified claim-level citations.
          </p>
        </section>
        <div className="learning-note">
          <Sparkles size={16} />
          <div>
            <strong>Made for learning</strong>
            <p>
              This fixed workflow retrieves evidence before generating a
              response.
            </p>
          </div>
        </div>
      </aside>
    </div>
  );
}

function EmptyChat({ onPickExample }: { onPickExample: (q: string) => void }) {
  return (
    <div className="empty-chat">
      <div className="empty-icon">
        <Sparkles size={24} />
      </div>
      <h2>What would you like to know?</h2>
      <p>Ask a question and explore the evidence behind the answer.</p>
      <div className="example-grid">
        {examples.map((q, i) => (
          <button key={q} onClick={() => onPickExample(q)}>
            <span className="example-num">0{i + 1}</span>
            <span>{q}</span>
            <ArrowRight size={15} />
          </button>
        ))}
      </div>
    </div>
  );
}

function RunMessage({
  run,
  selected,
  onSelect,
  onViewChunk,
}: {
  run: Run;
  selected: boolean;
  onSelect: () => void;
  onViewChunk: (id: number, sourceIndex?: Index) => void;
}) {
  return (
    <div
      className={"message-pair " + (selected ? "selected-run" : "")}
      onClick={onSelect}
    >
      <div className="message user-message">
        <span className="message-avatar user">Y</span>
        <div>
          <div className="message-head">
            You <span>QUESTION</span>
          </div>
          <p>{run.question}</p>
        </div>
      </div>
      <div className="message">
        <span className="message-avatar bot">
          <Sparkles size={15} />
        </span>
        <div>
          <div className="message-head">
            Rules Lab{" "}
            <span>
              {run.duration_ms
                ? `${(run.duration_ms / 1000).toFixed(1)}s`
                : "PROCESSING"}
            </span>
          </div>
          <div className="answer">
            <ReactMarkdown>
              {run.answer || run.error || "Searching the regulations…"}
            </ReactMarkdown>
          </div>
          {run.hits.filter((h) => h.accepted).length > 0 && (
            <div className="source-chips">
              {run.hits
                .filter((h) => h.accepted)
                .map((h) => (
                  <button
                    key={h.document.metadata.chunk_id}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelect();
                      onViewChunk(h.document.metadata.chunk_id, run.index);
                    }}
                  >
                    <FileText size={12} /> Chunk {h.document.metadata.chunk_id}
                  </button>
                ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SourceRow({ hit, onClick }: { hit: Hit; onClick: () => void }) {
  return (
    <button className="source-row" onClick={onClick}>
      <span className="source-icon">
        <FileText size={15} />
      </span>
      <span>
        <strong>{hit.document.metadata.section || "General"}</strong>
        <small>
          Chunk {hit.document.metadata.chunk_id} · {hit.score.toFixed(3)}{" "}
          relevance
        </small>
      </span>
      <ChevronRight size={15} />
    </button>
  );
}
