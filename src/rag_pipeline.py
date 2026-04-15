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
        max_new_tokens=80,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer

def answer_question(vector_store, llm, question):
    docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    context = context[:800]

    # Special handling for "what is this about"
    if "what is" in question.lower() and "pdf" in question.lower():
        answer = summarize_context(context)
    else:
        answer = summarize_context(context)

    return answer, docs

def summarize_context(context: str) -> str:
    lines = context.split("\n")

    # pick important-looking lines
    important = []
    for line in lines:
        line = line.strip()
        if len(line) > 20:
            important.append(line)

    # return first 2–3 meaningful lines
    return " ".join(important[:3])
    
def render_answer_with_sources(answer: str, docs: List) -> str:
    return f"{answer}\n\nSources:\n{format_sources(docs)}"
