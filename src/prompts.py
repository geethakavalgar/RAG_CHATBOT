SYSTEM_PROMPT = '''
You are a helpful assistant answering questions using only the provided document context.
If the answer is not supported by the context, say that the answer is not available in the uploaded documents.
Keep the answer clear and concise, then list source snippets.
'''.strip()

def build_prompt(context: str, question: str) -> str:
    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
"""
