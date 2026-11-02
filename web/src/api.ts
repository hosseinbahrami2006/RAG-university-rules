export async function api<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(
    "/api" + path,
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  if (!res.ok) {
    let msg = await res.text();
    try {
      const parsed = JSON.parse(msg);
      msg =
        typeof parsed.detail === "string"
          ? parsed.detail
          : JSON.stringify(parsed.detail);
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

export async function stream(
  path: string,
  body: unknown,
  onEvent: (event: any) => void,
) {
  const response = await fetch("/api" + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    let data = await response.json();
    throw Error(
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail),
    );
  }
  if (!response.body) throw Error("No response stream");
  const reader = response.body.getReader(),
    decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    let parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data: "));
      if (line) onEvent(JSON.parse(line.slice(6)));
    }
    if (done) break;
  }
}
