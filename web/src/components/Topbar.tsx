import { ChevronRight } from "lucide-react";
import { titles } from "../constants";
import type { PageId, Status } from "../types";

type Props = {
  page: PageId;
  status: Status | null;
};

export function Topbar({ page, status }: Props) {
  return (
    <header className="topbar">
      <div className="breadcrumb">
        WORKSPACE <ChevronRight size={13} />{" "}
        <span>{titles[page][0].toUpperCase()}</span>
      </div>
      <div className="top-right">
        <span className="status-badge">
          <span
            className={"status-dot " + (status?.indexed ? "green" : "amber")}
          ></span>
          {status?.indexed ? "Index ready" : "Index required"}
        </span>
        <div className="avatar">R</div>
      </div>
    </header>
  );
}
