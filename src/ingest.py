from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL_NAME

def load_and_split_pdfs(file_paths: List[str]):
    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    for path in file_paths:
        loader = PyPDFLoader(path)
        pdf_docs = loader.load()
        chunks = splitter.split_documents(pdf_docs)
        docs.extend(chunks)
    return docs

def build_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.from_documents(documents, embeddings)
