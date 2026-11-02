import { FileText } from "lucide-react";
import { titles } from "../constants";
import type { PageId } from "../types";

type Props = {
  page: PageId;
  onViewSource: () => void;
};

export function PageHeading({ page, onViewSource }: Props) {
  return (
    <div className="heading">
      <div>
        <div className="eyebrow">
          <span></span> UNIVERSITY RULES RAG
        </div>
        <h1>{titles[page][0]}</h1>
        <p>{titles[page][1]}</p>
      </div>
      <div className="heading-actions">
        <button className="ghost-button" onClick={onViewSource}>
          <FileText size={15} /> View source
        </button>
      </div>
    </div>
  );
}
