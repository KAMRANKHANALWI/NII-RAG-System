import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# === Load environment variables from .env ===
load_dotenv()

# ==== Configuration Section ====
"""
Global configuration for the RAG system including:
- Vector database directory path
- Embedding model for semantic similarity
- LLM model for generating responses
"""
VECTOR_DB_DIR = "../vectorstores"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(
    model="llama3-8b-8192", temperature=0.1, api_key=os.getenv("GROQ_API_KEY")
)
