from typing import List, Tuple

from .config import TOP_K, MAX_NEW_TOKENS
from .prompts import build_prompt
from .utils import format_sources


def generate_answer(tokenizer, model, prompt: str) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.15
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer


def answer_question(vector_store, llm, question: str) -> Tuple[str, List]:
    tokenizer, model = llm
    docs = vector_store.similarity_search(question, k=TOP_K)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = build_prompt(context=context, question=question)
    answer = generate_answer(tokenizer, model, prompt)
    return answer, docs


def render_answer_with_sources(answer: str, docs: List) -> str:
    return f"{answer}\n\nSources:\n{format_sources(docs)}"
