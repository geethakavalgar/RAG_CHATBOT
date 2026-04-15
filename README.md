# RAG PDF Chatbot Pro

A production-style starter project for a Retrieval-Augmented Generation (RAG) chatbot that:
- accepts multiple PDF uploads
- chunks documents
- embeds content with Sentence Transformers
- stores embeddings in FAISS
- retrieves relevant passages
- answers grounded questions
- shows source citations
- maintains chat history in the UI

## Features
- Multiple PDF support
- Source-aware responses
- Streamlit chat interface
- Simple session memory
- Modular Python code
- Docker-ready
- Streamlit Community Cloud ready

## Project Structure
```bash
rag_pdf_chatbot_pro/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── llm.py
│   ├── prompts.py
│   ├── rag_pipeline.py
│   └── utils.py
├── tests/
│   └── test_utils.py
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Docker
```bash
docker build -t rag-pdf-chatbot-pro .
docker run -p 8501:8501 rag-pdf-chatbot-pro
```

## Notes
- This starter uses local Hugging Face embeddings and a lightweight text generation pipeline.
- For stronger answers, replace the local generator with an API-backed LLM later.
- Recommended next upgrade: add persistent vector store and authentication.

## Resume Bullet
Developed a production-style RAG chatbot that ingests multiple PDFs, performs semantic retrieval using Sentence Transformers and FAISS, and generates grounded responses with source citations through a Streamlit interface.

## Sample Results

### Question
What is the uploaded PDF about?

### Answer
The uploaded PDF appears to be a receipt or payment-related document.

### Screenshot
![App Screenshot](assets/rag-demo.mov)
