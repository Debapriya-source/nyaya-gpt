"""Vectorless RAG using LLM-guided tree navigation.

Parses legal PDFs into hierarchical trees, then uses the LLM to reason
about which chapters/sections are relevant — achieving relevance-based
retrieval instead of similarity-based retrieval.
"""

import json
import re

from langchain.agents import tool

from app.services.tree_index import TreeNode, find_node, get_node_content


def _parse_id_list(text: str) -> list[str]:
    """Extract a JSON array of string items from an LLM response."""
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return [str(item).strip() for item in json.loads(text[start:end])]
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _resolve_ids(raw_items: list[str], children: list[TreeNode]) -> list[str]:
    """Map raw LLM items (node IDs, bare numbers, Roman numerals) to actual node IDs."""
    resolved = []
    for item in raw_items:
        if "." in item:
            resolved.append(item)
            continue
        for child in children:
            if item in child.id or item.lower() in child.id:
                resolved.append(child.id)
                break
    return resolved


def retrieve_by_tree(query: str, tree: TreeNode, llm) -> str:
    """Two-level LLM-guided tree navigation: pick chapters, then sections."""

    # Step 1: Pick relevant chapters/parts
    chapter_lines = []
    for child in tree.children:
        n = f" ({len(child.children)} sections)" if child.children else ""
        chapter_lines.append(f"- [{child.id}] {child.title}{n}")

    example = tree.children[0].id if tree.children else "id"
    prompt1 = (
        f"Pick the most relevant chapters/parts for this legal query.\n\n"
        f"Query: {query}\n\nChapters:\n" + "\n".join(chapter_lines) +
        f'\n\nReturn ONLY a JSON array of 1-3 IDs from the list above (e.g. ["{example}"]).'
    )
    resp1 = llm.invoke(prompt1)
    content1 = resp1.content if hasattr(resp1, "content") else str(resp1)

    chapter_ids = _resolve_ids(_parse_id_list(content1), tree.children)

    # Fallback: extract full IDs via regex
    if not chapter_ids:
        pattern = rf'["\']?({re.escape(tree.id)}\.[a-z0-9_.]+)["\']?'
        chapter_ids = re.findall(pattern, content1, re.IGNORECASE)
    # Last resort: keyword match chapter titles
    if not chapter_ids:
        query_lower = query.lower()
        for child in tree.children:
            if any(w in child.title.lower() for w in query_lower.split()):
                chapter_ids.append(child.id)
        if not chapter_ids:
            chapter_ids = [tree.children[0].id] if tree.children else []

    # Step 2: Pick sections within selected chapters
    section_lines = []
    available_sections: set[str] = set()
    for ch_id in chapter_ids[:3]:
        node = find_node(tree, ch_id)
        if node:
            for child in node.children:
                section_lines.append(f"- [{child.id}] {child.title}")
                available_sections.add(child.id)

    if not section_lines:
        parts = [get_node_content(tree, ch_id) for ch_id in chapter_ids[:3]]
        return "\n\n---\n\n".join(p for p in parts if p)

    sec_example = next(iter(available_sections), "id")
    prompt2 = (
        f"Pick the most relevant sections for this legal query.\n\n"
        f"Query: {query}\n\nSections:\n" + "\n".join(section_lines) +
        f'\n\nReturn ONLY a JSON array of 1-5 IDs from the list above (e.g. ["{sec_example}"]).'
    )
    resp2 = llm.invoke(prompt2)
    content2 = resp2.content if hasattr(resp2, "content") else str(resp2)

    raw_ids = _parse_id_list(content2)
    section_ids = []
    for item in raw_ids:
        if item in available_sections:
            section_ids.append(item)
        elif item.isdigit():
            for avail in available_sections:
                if f"sec_{item}" in avail:
                    section_ids.append(avail)
                    break
    if not section_ids:
        section_ids = list(available_sections)[:5]

    parts = [get_node_content(tree, sid) for sid in section_ids[:5]]
    return "\n\n---\n\n".join(p for p in parts if p) or "No relevant sections found."


def create_tool_for_tree(tree: TreeNode, llm, name: str, description: str):
    """Create a LangChain tool that uses tree-based retrieval."""

    @tool(name, description=description)
    def query_tool(query: str) -> str:
        """Query the document via LLM-guided tree navigation."""
        return retrieve_by_tree(query, tree, llm)

    return query_tool
