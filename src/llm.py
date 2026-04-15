from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .config import GENERATION_MODEL_NAME


def get_llm():
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)
    return tokenizer, model
