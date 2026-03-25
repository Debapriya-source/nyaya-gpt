const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/** Payload for sending a chat message to the backend. */
export interface ChatRequest {
  message: string;
  provider: "groq" | "ollama";
  ollama_model?: string;
}

/** Backend response describing available LLM providers. */
export interface ProviderStatus {
  ollama_available: boolean;
  ollama_models: string[];
}

/** Fetch the current LLM provider status from the backend. */
export async function getProviderStatus(): Promise<ProviderStatus> {
  const res = await fetch(`${API_URL}/api/providers/status`);
  if (!res.ok) throw new Error("Failed to fetch provider status");
  return res.json();
}

/** Send a chat message and return a readable stream reader for SSE tokens. */
export async function sendMessage(
  req: ChatRequest,
): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error("Failed to send message");
  if (!res.body) throw new Error("No response body");
  return res.body.getReader();
}
