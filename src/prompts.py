SYSTEM_PROMPT = """
You are a helpful assistant.
Answer using only the provided context.
If the answer is not in the context, say:
"The answer is not available in the uploaded documents."
Keep the answer short and factual.
""".strip()


def build_prompt(context: str, question: str) -> str:
    return f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
""".strip()
