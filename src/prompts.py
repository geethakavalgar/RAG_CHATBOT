prompt = f"""
Look at the document text and answer the question.

Explain clearly what the document is about in one short sentence.

Be specific. Do NOT just say "document".
Use important details like names, amounts, or purpose.

Document text:
{context}

Question:
{question}

Answer:
""".strip()
