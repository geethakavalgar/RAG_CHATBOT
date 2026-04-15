from typing import List, Tuple

from .config import TOP_K
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
        max_new_tokens=80,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer


def generic_document_summary(context: str) -> str:
    text = context.lower()

    if "receipt" in text:
        return "The uploaded PDF appears to be a receipt or payment-related document."
    if "invoice" in text:
        return "The uploaded PDF appears to be an invoice or billing document."
    if "report" in text:
        return "The uploaded PDF appears to be a report."
    if "donation" in text:
        return "The uploaded PDF appears to be a donation-related document."
    if "research" in text or "abstract" in text or "introduction" in text:
        return "The uploaded PDF appears to be a research or academic document."

    clean = " ".join(context.split())
    return f"The uploaded PDF appears to be a document about: {clean[:120]}..."


def answer_question(vector_store, llm, question: str) -> Tuple[str, List]:
    tokenizer, model = llm
    docs = vector_store.similarity_search(question, k=TOP_K)

    context = "\n\n".join(doc.page_content for doc in docs)
    context = " ".join(context.split())
    context = context[:700]

    q = question.lower().strip()

    if "what is the uploaded pdf about" in q or "what is this pdf about" in q:
        answer = generic_document_summary(context)
    else:
        prompt = f"""
Answer the question using the document text below.
Be short, clear, and factual.
Do not invent anything.

Document text:
{context}

Question:
{question}

Answer:
""".strip()

        answer = generate_answer(tokenizer, model, prompt)

        if not answer or len(answer.strip()) < 5 or answer.strip().lower() in ["document", "unclear"]:
            answer = generic_document_summary(context)

    return answer, docs


def render_answer_with_sources(answer: str, docs: List) -> str:
    return f"{answer}\n\nSources:\n{format_sources(docs)}"
