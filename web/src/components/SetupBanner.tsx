import { Database, LoaderCircle } from "lucide-react";

type Props = {
  busy: boolean;
  disabled: boolean;
  indexProgress: string;
  onIndex: () => void;
};

export function SetupBanner({
  busy,
  disabled,
  indexProgress,
  onIndex,
}: Props) {
  return (
    <div className="setup-banner">
      <div className="setup-icon">
        <Database size={19} />
      </div>
      <div>
        <strong>Build your knowledge index</strong>
        <p>
          Turn the university regulations into searchable chunks to start
          exploring.
        </p>
        {indexProgress && <small>{indexProgress}</small>}
      </div>
      <button
        disabled={busy || disabled}
        className="primary small"
        onClick={onIndex}
      >
        {busy ? (
          <LoaderCircle className="spin" size={15} />
        ) : (
          <Database size={15} />
        )}{" "}
        Build index
      </button>
    </div>
  );
}
