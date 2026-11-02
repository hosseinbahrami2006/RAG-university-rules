import { X } from "lucide-react";

type Props = {
  message: string;
  onDismiss: () => void;
};

export function Notice({ message, onDismiss }: Props) {
  return (
    <div className="notice">
      <span>{message}</span>
      <button onClick={onDismiss} aria-label="Dismiss">
        <X size={16} />
      </button>
    </div>
  );
}
