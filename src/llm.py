from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

from .config import GENERATION_MODEL_NAME, MAX_NEW_TOKENS

def get_llm():
    generator = pipeline(
        "text-generation",
        model=GENERATION_MODEL_NAME,
        max_new_tokens=MAX_NEW_TOKENS
    )
    return HuggingFacePipeline(pipeline=generator)
