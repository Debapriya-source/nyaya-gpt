"use client";

import { useEffect, useState } from "react";
import { getProviderStatus } from "@/lib/api";

/** Props for the Sidebar component. */
interface SidebarProps {
  open: boolean;
  onClose: () => void;
  provider: "groq" | "ollama";
  onProviderChange: (p: "groq" | "ollama") => void;
  ollamaModel: string;
  onOllamaModelChange: (m: string) => void;
  onNewChat: () => void;
}

/** Sidebar panel for provider selection, model config, and new-chat action. */
export default function Sidebar({
  open,
  onClose,
  provider,
  onProviderChange,
  ollamaModel,
  onOllamaModelChange,
  onNewChat,
}: SidebarProps) {
  const [ollamaAvailable, setOllamaAvailable] = useState(false);
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);

  useEffect(() => {
    getProviderStatus()
      .then((status) => {
        setOllamaAvailable(status.ollama_available);
        setOllamaModels(status.ollama_models);
      })
      .catch(() => {
        setOllamaAvailable(false);
      });
  }, []);

  return (
    <>
      {/* Overlay for mobile */}
      {open && (
        <div
          className="fixed inset-0 z-30 bg-black/30 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed left-0 top-0 z-40 flex h-full w-64 flex-col border-r border-zinc-200 bg-zinc-50 transition-transform dark:border-zinc-800 dark:bg-zinc-900 lg:static lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3 dark:border-zinc-800">
          <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
            Settings
          </span>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-zinc-400 hover:text-zinc-600 lg:hidden"
          >
            &#10005;
          </button>
        </div>

        <div className="flex flex-1 flex-col gap-6 overflow-y-auto p-4">
          {/* New Chat */}
          <button
            onClick={onNewChat}
            className="w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-700 transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700"
          >
            + New Chat
          </button>

          {/* Provider Selection */}
          <div>
            <label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
              LLM Provider
            </label>
            <div className="flex flex-col gap-2">
              <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-zinc-200 px-3 py-2 transition-colors hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800">
                <input
                  type="radio"
                  name="provider"
                  value="groq"
                  checked={provider === "groq"}
                  onChange={() => onProviderChange("groq")}
                  className="accent-amber-600"
                />
                <div>
                  <span className="text-sm font-medium text-zinc-800 dark:text-zinc-200">
                    Groq
                  </span>
                  <span className="ml-1 text-xs text-zinc-500">(Cloud)</span>
                </div>
              </label>
              <label
                className={`flex cursor-pointer items-center gap-2 rounded-lg border px-3 py-2 transition-colors ${
                  ollamaAvailable
                    ? "border-zinc-200 hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                    : "cursor-not-allowed border-zinc-100 opacity-50 dark:border-zinc-800"
                }`}
              >
                <input
                  type="radio"
                  name="provider"
                  value="ollama"
                  checked={provider === "ollama"}
                  onChange={() => onProviderChange("ollama")}
                  disabled={!ollamaAvailable}
                  className="accent-amber-600"
                />
                <div>
                  <span className="text-sm font-medium text-zinc-800 dark:text-zinc-200">
                    OLLAMA
                  </span>
                  <span className="ml-1 text-xs text-zinc-500">(Local)</span>
                </div>
              </label>
            </div>

            {!ollamaAvailable && (
              <p className="mt-1 text-xs text-zinc-400">
                OLLAMA not detected
              </p>
            )}
          </div>

          {/* OLLAMA Model Select */}
          {provider === "ollama" && ollamaAvailable && (
            <div>
              <label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                Model
              </label>
              <select
                value={ollamaModel}
                onChange={(e) => onOllamaModelChange(e.target.value)}
                className="w-full rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-800 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200"
              >
                {ollamaModels.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Current Config */}
          <div className="mt-auto rounded-lg bg-zinc-100 p-3 dark:bg-zinc-800">
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              <span className="font-medium">Provider:</span>{" "}
              {provider === "groq" ? "Groq (Cloud)" : "OLLAMA (Local)"}
            </p>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              <span className="font-medium">Model:</span>{" "}
              {provider === "groq" ? "llama3-8b-8192" : ollamaModel}
            </p>
          </div>
        </div>
      </aside>
    </>
  );
}
