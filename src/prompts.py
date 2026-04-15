def build_prompt(context: str, question: str) -> str:
    return f"""
Answer the question using the context below.

IMPORTANT:
- Do NOT copy the text directly
- Write the answer in your own words
- Keep it short

Context:
{context}

Question:
{question}

Answer:
"""
