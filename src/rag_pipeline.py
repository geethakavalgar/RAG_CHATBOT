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
        max_new_tokens=60,
        do_sample=False,
        repetition_penalty=1.25,
        no_repeat_ngram_size=3
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer


def answer_question(vector_store, llm, question: str) -> Tuple[str, List]:
    tokenizer, model = llm
    docs = vector_store.similarity_search(question, k=TOP_K)

    context = "\n\n".join(doc.page_content for doc in docs)
    context = " ".join(context.split())
    context = context[:700]

    prompt = f"""
Read the document text and answer the question briefly.

Instructions:
- Answer in one short sentence.
- Use your own words.
- Do not repeat the document text.
- Do not invent details.
- If the question asks what the PDF is about, say what kind of document it is and its main purpose.
- If the answer cannot be determined, say: The document content is unclear.

Document text:
{context}

Question:
{question}

Short answer:
""".strip()

    answer = generate_answer(tokenizer, model, prompt)

    if not answer or len(answer) < 5:
        answer = "The document content is unclear."

    return answer, docs


def render_answer_with_sources(answer: str, docs: List) -> str:
    return f"{answer}\n\nSources:\n{format_sources(docs)}"
