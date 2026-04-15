def build_prompt(context: str, question: str) -> str:
    return f"""
You are an AI assistant.

Your job is to READ the context and ANSWER the question.

Rules:
- Do NOT copy the context
- Give a SHORT summary answer
- Use your own words

Context:
{context}

Question:
{question}

Final Answer:
"""
