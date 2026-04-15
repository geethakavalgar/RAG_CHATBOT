from typing import List

def clean_text(text: str) -> str:
    return " ".join(text.split())

def format_sources(docs: List) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source = meta.get("source", "Unknown source")
        page = meta.get("page", "N/A")
        snippet = clean_text(doc.page_content)[:220]
        lines.append(f"[{i}] {source} | page {page} | {snippet}...")
    return "\n".join(lines)
