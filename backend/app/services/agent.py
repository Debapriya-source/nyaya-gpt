"""ReAct agent orchestration for legal queries."""

import json
import os
import warnings
from collections.abc import AsyncGenerator

from langchain.agents import AgentExecutor, create_react_agent
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from app.prompts.react_template import get_prompt_template
from app.services.rag import create_tool_for_tree
from app.services.tree_index import TreeNode


def _create_llm(
    provider: str,
    ollama_model: str | None = None,
):
    """Instantiate the LLM for the given provider."""
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=ollama_model or "llama3.2:3b",
            base_url=base_url,
            temperature=0.1,
            timeout=120,
        )
    return ChatGroq(model="llama3-8b-8192")


def _build_executor(
    provider: str,
    ollama_model: str | None,
    constitution_tree: TreeNode,
    bns_tree: TreeNode,
) -> AgentExecutor:
    """Build a ReAct AgentExecutor with legal RAG tools and the selected LLM."""
    warnings.filterwarnings("ignore", category=FutureWarning)

    llm = _create_llm(provider, ollama_model)

    tools = [
        create_tool_for_tree(
            constitution_tree,
            llm,
            "indian_constitution_query",
            "Search the Indian Constitution using intelligent tree-based retrieval. Use for questions about constitutional rights, government structure, and fundamental duties.",
        ),
        create_tool_for_tree(
            bns_tree,
            llm,
            "indian_laws_query",
            "Search the Bharatiya Nyaya Sanhita 2023 (Indian criminal law) using intelligent tree-based retrieval. Use for questions about crimes, punishments, and criminal offences.",
        ),
    ]

    prompt_template = get_prompt_template()
    react_agent = create_react_agent(llm, tools, prompt_template)

    return AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=20,
    )


async def run_agent_stream(
    query: str,
    provider: str,
    ollama_model: str | None,
    constitution_tree: TreeNode,
    bns_tree: TreeNode,
) -> AsyncGenerator[str, None]:
    """Stream ReAct agent output as SSE-formatted JSON lines."""
    try:
        executor = _build_executor(provider, ollama_model, constitution_tree, bns_tree)
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return
    try:
        final_output = ""
        async for event in executor.astream_events({"input": query}, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token and isinstance(token, str):
                    yield f"data: {json.dumps({'token': token})}\n\n"
            elif kind == "on_chain_end" and event["name"] == "AgentExecutor":
                output = event["data"].get("output", "")
                if isinstance(output, dict):
                    final_output = output.get("output", str(output))
                else:
                    final_output = str(output) if output else ""

        if final_output:
            yield f"data: {json.dumps({'done': True, 'output': final_output})}\n\n"
        else:
            yield f"data: {json.dumps({'done': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
