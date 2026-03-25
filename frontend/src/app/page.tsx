"use client";

import { useState } from "react";
import Header from "@/components/Header";
import Sidebar from "@/components/Sidebar";
import ChatWindow from "@/components/ChatWindow";
import ChatInput from "@/components/ChatInput";
import { useChat } from "@/hooks/useChat";

/** Main page composing sidebar, header, chat window, and input. */
export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [provider, setProvider] = useState<"groq" | "ollama">("groq");
  const [ollamaModel, setOllamaModel] = useState("llama3.2:3b");

  const { messages, isStreaming, send, clearMessages } = useChat();

  /** Send a message using the currently selected provider and model. */
  function handleSend(content: string) {
    send(content, provider, provider === "ollama" ? ollamaModel : undefined);
  }

  /** Clear chat history and close the sidebar. */
  function handleNewChat() {
    clearMessages();
    setSidebarOpen(false);
  }

  return (
    <div className="flex h-full">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        provider={provider}
        onProviderChange={setProvider}
        ollamaModel={ollamaModel}
        onOllamaModelChange={setOllamaModel}
        onNewChat={handleNewChat}
      />

      <div className="flex flex-1 flex-col">
        <Header onToggleSidebar={() => setSidebarOpen(!sidebarOpen)} />
        <ChatWindow messages={messages} isStreaming={isStreaming} />
        <ChatInput onSend={handleSend} disabled={isStreaming} />
      </div>
    </div>
  );
}
