"use client";

import { useCallback, useState } from "react";
import { sendMessage } from "@/lib/api";

/** A single chat message (user or assistant). */
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

/** Hook managing chat messages, streaming state, and send/clear actions. */
export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  const send = useCallback(
    async (
      content: string,
      provider: "groq" | "ollama",
      ollamaModel?: string,
    ) => {
      if (!content.trim() || isStreaming) return;

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: content.trim(),
      };
      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "",
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);

      const updateAssistant = (text: string) =>
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id ? { ...m, content: text } : m,
          ),
        );

      try {
        const reader = await sendMessage({
          message: content.trim(),
          provider,
          ollama_model: ollamaModel,
        });

        const decoder = new TextDecoder();
        let buffer = "";
        let finalOutput = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          let updated = false;
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
              const data = JSON.parse(line.slice(6));
              if (data.error) {
                finalOutput = `Error: ${data.error}`;
                updated = true;
              } else if (data.done && data.output) {
                finalOutput = data.output;
                updated = true;
              } else if (data.token) {
                finalOutput += data.token;
                updated = true;
              }
            } catch {
              // skip malformed JSON
            }
          }

          // Batch: update UI once per read() chunk, not per SSE line
          if (updated) {
            updateAssistant(finalOutput);
          }
        }

        if (!finalOutput) {
          updateAssistant(
            "Sorry, I could not generate a response. Please try again.",
          );
        }
      } catch (err) {
        updateAssistant(
          `Connection error: ${err instanceof Error ? err.message : "Unknown error"}`,
        );
      } finally {
        setIsStreaming(false);
      }
    },
    [isStreaming],
  );

  return { messages, isStreaming, send, clearMessages };
}
