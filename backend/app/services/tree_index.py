"""Build hierarchical tree indexes from legal PDF documents.

Instead of embedding chunks into vectors, this module parses PDFs into
structured trees that mirror the document's natural hierarchy
(Parts/Chapters -> Articles/Sections). An LLM then reasons over the tree
outline to find relevant nodes — achieving relevance-based retrieval
instead of similarity-based retrieval.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PyPDF2 import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_DIR = BASE_DIR / "db"
DATA_DIR = BASE_DIR / "data"

_PAGE_MARKER_RE = re.compile(r"<<<PAGE:(\d+)>>>")


def _inject_page_markers(pages: list[tuple[int, str]]) -> str:
    """Join page texts with embedded page-number markers."""
    return "".join(f"\n<<<PAGE:{n}>>>\n{t}" for n, t in pages)


def _page_range(text: str, fallback: int = 0) -> tuple[int, int]:
    """Extract (start_page, end_page) from page markers in text."""
    nums = [int(m) for m in _PAGE_MARKER_RE.findall(text)]
    return (nums[0], nums[-1]) if nums else (fallback, fallback)


def _strip_markers(text: str) -> str:
    """Remove page markers and strip whitespace."""
    return _PAGE_MARKER_RE.sub("", text).strip()


@dataclass
class TreeNode:
    """A node in the hierarchical document tree (root, part/chapter, article/section)."""
    id: str
    title: str
    summary: str
    level: int
    page_start: int
    page_end: int
    content: str
    children: list[TreeNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize the tree node and its children to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TreeNode:
        """Deserialize a dictionary (including nested children) into a TreeNode."""
        children = [cls.from_dict(c) for c in data.get("children", [])]
        return cls(
            id=data["id"],
            title=data["title"],
            summary=data["summary"],
            level=data["level"],
            page_start=data["page_start"],
            page_end=data["page_end"],
            content=data["content"],
            children=children,
        )


def _extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text from each page, returning (page_number, text) pairs."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append((i, text))
    return pages


def _build_constitution_tree(pages: list[tuple[int, str]]) -> TreeNode:
    """Parse the Indian Constitution PDF into a Part -> Article tree."""
    full_text_with_markers = _inject_page_markers(pages)

    # Extract Part boundaries
    part_pattern = re.compile(
        r"PART\s+([IVXLCDM]+[A-Z]*)\s*\n\s*(.+?)(?:\n|$)", re.MULTILINE
    )
    article_pattern = re.compile(
        r"(\d+[A-Z]?)\.\s*(.+?)(?:\.\s*[—\-]|\.\s*$|\s*[—\-])", re.MULTILINE
    )

    root = TreeNode(
        id="constitution",
        title="The Constitution of India",
        summary="The supreme law of India, containing fundamental rights, directive principles, and the structure of government.",
        level=0,
        page_start=0,
        page_end=len(pages) - 1,
        content="",
        children=[],
    )

    # Split into parts
    part_matches = list(part_pattern.finditer(full_text_with_markers))

    for i, match in enumerate(part_matches):
        part_num = match.group(1).strip()
        part_title = match.group(2).strip()

        start_pos = match.start()
        end_pos = (
            part_matches[i + 1].start() if i + 1 < len(part_matches) else len(full_text_with_markers)
        )
        part_text = full_text_with_markers[start_pos:end_pos]

        p_start, p_end = _page_range(part_text)

        part_id = f"constitution.part_{part_num.lower()}"
        part_node = TreeNode(
            id=part_id,
            title=f"Part {part_num}: {part_title}",
            summary=part_title,
            level=1,
            page_start=p_start,
            page_end=p_end,
            content="",
            children=[],
        )

        # Extract articles within this part
        article_matches = list(article_pattern.finditer(part_text))
        for j, art_match in enumerate(article_matches):
            art_num = art_match.group(1).strip()
            art_title = art_match.group(2).strip()

            art_start = art_match.start()
            art_end = (
                article_matches[j + 1].start()
                if j + 1 < len(article_matches)
                else len(part_text)
            )
            art_slice = part_text[art_start:art_end]
            art_content = _strip_markers(art_slice)
            a_start, a_end = _page_range(art_slice, p_start)

            art_id = f"{part_id}.art_{art_num.lower()}"
            art_node = TreeNode(
                id=art_id,
                title=f"Article {art_num}: {art_title}",
                summary=art_title,
                level=2,
                page_start=a_start,
                page_end=a_end,
                content=art_content[:3000],  # cap content size
                children=[],
            )
            part_node.children.append(art_node)

        # If no articles found, store part text as content
        if not part_node.children:
            part_node.content = _strip_markers(part_text)[:3000]

        root.children.append(part_node)

    return root


def _build_bns_tree(pages: list[tuple[int, str]]) -> TreeNode:
    """Parse the BNS PDF into a Chapter -> Section tree."""
    full_text_with_markers = _inject_page_markers(pages)

    chapter_pattern = re.compile(
        r"CHAPTER\s+([IVXLCDM]+)\s*\n\s*(.+?)(?:\n\d+\.\s|\n<<<PAGE)", re.DOTALL
    )
    section_title_pattern = re.compile(
        r"(\d+)\.\s+(.+?)(?:\.\s*$|\.\s*\n)", re.MULTILINE
    )

    root = TreeNode(
        id="bns",
        title="The Bharatiya Nyaya (Second) Sanhita, 2023",
        summary="Comprehensive Indian criminal law code replacing the Indian Penal Code. Covers offences, punishments, and criminal liability.",
        level=0,
        page_start=0,
        page_end=len(pages) - 1,
        content="",
        children=[],
    )

    # Build a title lookup from the TOC section (first ~15 pages)
    toc_text = ""
    for page_num, text in pages[:15]:
        toc_text += text + "\n"
    section_titles: dict[str, str] = {}
    for m in section_title_pattern.finditer(toc_text):
        section_titles[m.group(1).strip()] = m.group(2).strip()

    chapter_matches = list(chapter_pattern.finditer(full_text_with_markers))

    # Deduplicate chapters: keep the LAST occurrence of each chapter number
    # (first occurrence is typically in the TOC, last is in actual content)
    chapter_by_num: dict[str, re.Match] = {}
    for m in chapter_matches:
        ch_num = m.group(1).strip()
        chapter_by_num[ch_num] = m  # last wins
    chapter_matches = sorted(chapter_by_num.values(), key=lambda m: m.start())

    for i, match in enumerate(chapter_matches):
        ch_num = match.group(1).strip()
        ch_title = _clean_title(match.group(2))

        start_pos = match.start()
        end_pos = (
            chapter_matches[i + 1].start()
            if i + 1 < len(chapter_matches)
            else len(full_text_with_markers)
        )
        ch_text = full_text_with_markers[start_pos:end_pos]

        p_start, p_end = _page_range(ch_text)

        ch_id = f"bns.ch_{ch_num.lower()}"
        ch_node = TreeNode(
            id=ch_id,
            title=f"Chapter {ch_num}: {ch_title}",
            summary=ch_title,
            level=1,
            page_start=p_start,
            page_end=p_end,
            content="",
            children=[],
        )

        # Find section bodies in this chapter's text
        # Use a simpler pattern: lines starting with a number followed by ". ("
        sec_splits = list(re.finditer(r"\n(\d{1,3})\.\s", ch_text))
        # Keep only unique section numbers, prefer later (content) over earlier (TOC)
        sec_by_num: dict[str, re.Match] = {}
        for sm in sec_splits:
            sn = sm.group(1)
            # Prefer matches with actual content (longer text until next section)
            sec_by_num[sn] = sm
        section_matches = list(sec_by_num.values())
        section_matches.sort(key=lambda m: m.start())

        for j, sec_match in enumerate(section_matches):
            sec_num = sec_match.group(1).strip()
            sec_title = section_titles.get(sec_num, f"Section {sec_num}")

            sec_start = sec_match.start()
            sec_end = (
                section_matches[j + 1].start()
                if j + 1 < len(section_matches)
                else len(ch_text)
            )
            sec_slice = ch_text[sec_start:sec_end]
            sec_content = _strip_markers(sec_slice)
            s_start, s_end = _page_range(sec_slice, p_start)

            sec_id = f"{ch_id}.sec_{sec_num}"
            sec_node = TreeNode(
                id=sec_id,
                title=f"Section {sec_num}: {sec_title}",
                summary=sec_title,
                level=2,
                page_start=s_start,
                page_end=s_end,
                content=sec_content[:3000],
                children=[],
            )
            ch_node.children.append(sec_node)

        if not ch_node.children:
            ch_node.content = _strip_markers(ch_text)[:3000]

        root.children.append(ch_node)

    return root


def _clean_title(title: str) -> str:
    """Clean PDF extraction artifacts from titles."""
    title = _PAGE_MARKER_RE.sub("", title)
    # Collapse multiple spaces (PDF spacing artifacts)
    title = re.sub(r"\s+", " ", title)
    # Remove trailing section numbers/content that leaked in
    title = re.sub(r"\d+\.\s*\w+.*$", "", title)
    # Remove "CLAUSES" and similar TOC artifacts
    title = re.sub(r"CLAUSES.*$", "", title, flags=re.IGNORECASE)
    # Clean "Of ..." subheadings that leaked into the title
    title = re.sub(r"\s+Of\s+.*$", "", title)
    return title.strip()


def build_tree(doc_type: str, pdf_path: str) -> TreeNode:
    """Build a hierarchical tree from a PDF based on the document type."""
    pages = _extract_pages(pdf_path)
    if doc_type == "constitution":
        return _build_constitution_tree(pages)
    elif doc_type == "bns":
        return _build_bns_tree(pages)
    else:
        raise ValueError(f"Unknown doc_type: {doc_type}")


def load_or_build_tree(doc_type: str, pdf_filename: str) -> TreeNode:
    """Load a cached tree index or build one from the PDF."""
    cache_path = DB_DIR / f"tree_index_{doc_type}.json"
    pdf_path = str(DATA_DIR / pdf_filename)

    if cache_path.exists():
        with open(cache_path) as f:
            return TreeNode.from_dict(json.load(f))

    tree = build_tree(doc_type, pdf_path)

    os.makedirs(str(DB_DIR), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(tree.to_dict(), f, indent=2)

    return tree


def find_node(tree: TreeNode, node_id: str) -> TreeNode | None:
    """Find a node by its ID in the tree."""
    if tree.id == node_id:
        return tree
    for child in tree.children:
        result = find_node(child, node_id)
        if result:
            return result
    return None


def get_node_content(tree: TreeNode, node_id: str) -> str:
    """Get the full content of a node and its children."""
    node = find_node(tree, node_id)
    if not node:
        return ""
    if node.content:
        return f"[{node.title}]\n{node.content}"
    # If node has no direct content, concatenate children
    parts = []
    for child in node.children:
        if child.content:
            parts.append(f"[{child.title}]\n{child.content}")
    return "\n\n".join(parts)
