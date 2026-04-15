from typing import List, Tuple

from .config import TOP_K
from .prompts import build_prompt
from .utils import format_sources

def answer_question(vector_store, llm, question: str) -> Tuple[str, List]:
    docs = vector_store.similarity_search(question, k=TOP_K)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = build_prompt(context=context, question=question)
    answer = llm.invoke(prompt)
    return answer, docs

def render_answer_with_sources(answer: str, docs: List) -> str:
    return f"{answer}\n\nSources:\n{format_sources(docs)}"
