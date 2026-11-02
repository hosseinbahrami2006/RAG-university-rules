import { ChevronRight, CircleHelp, Sparkles } from "lucide-react";
import { menu } from "../constants";
import type { PageId } from "../types";

type Props = {
  page: PageId;
  onNavigate: (page: PageId) => void;
};

export function Sidebar({ page, onNavigate }: Props) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          <Sparkles size={19} />
        </div>
        <div>
          <strong>Rules Lab</strong>
          <small>RAG EXPLORER</small>
        </div>
      </div>
      <div className="side-label">WORKSPACE</div>
      <nav>
        {menu.map(([id, label, Icon]) => (
          <button
            key={id}
            className={"nav-item " + (page === id ? "active" : "")}
            onClick={() => onNavigate(id)}
          >
            <Icon size={17} />
            <span>{label}</span>
            {page === id && <ChevronRight size={14} />}
          </button>
        ))}
      </nav>
      <div className="side-bottom">
        <div className="side-note">
          <span className="tiny-icon">
            <CircleHelp size={15} />
          </span>
          <strong>Learning workspace</strong>
          <p>A closer look at how retrieval-augmented generation works.</p>
        </div>
        <div className="side-footer">
          <span className="pulse"></span> Local environment{" "}
          <span className="version">v1.0</span>
        </div>
      </div>
    </aside>
  );
}
