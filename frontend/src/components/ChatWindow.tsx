"use client";

import { useEffect, useRef } from "react";
import type { Message } from "@/hooks/useChat";
import MessageBubble from "./MessageBubble";

/** Scrollable chat message list with a welcome screen when empty. */
export default function ChatWindow({
  messages,
  isStreaming,
}: {
  messages: Message[];
  isStreaming: boolean;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8 text-center">
        <span className="text-5xl">&#9878;</span>
        <h2 className="text-xl font-semibold text-zinc-800 dark:text-zinc-200">
          Welcome to Nyaya-GPT
        </h2>
        <p className="max-w-md text-sm leading-relaxed text-zinc-500 dark:text-zinc-400">
          Your AI assistant for Indian legal documents. Ask questions about the
          Indian Constitution or Bharatiya Nyaya Sanhita (BNS).
        </p>
        <div className="mt-2 flex flex-wrap justify-center gap-2">
          {[
            "What are fundamental rights?",
            "Explain Article 21",
            "What is the punishment for theft under BNS?",
          ].map((q) => (
            <span
              key={q}
              className="rounded-full border border-zinc-200 px-3 py-1.5 text-xs text-zinc-600 dark:border-zinc-700 dark:text-zinc-400"
            >
              {q}
            </span>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col gap-4 overflow-y-auto p-4">
      {messages.map((msg, i) => (
        <MessageBubble
          key={msg.id}
          message={msg}
          isStreaming={
            isStreaming &&
            msg.role === "assistant" &&
            i === messages.length - 1
          }
        />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
