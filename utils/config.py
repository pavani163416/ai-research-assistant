import os

# Text Chunking Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))

# Vector Store Settings
TOP_K = int(os.getenv("TOP_K", 5))

# Models Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Storage Paths
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "documents/uploads")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "vectorstore/storage/index.faiss")
FAISS_CHUNKS_PATH = os.getenv("FAISS_CHUNKS_PATH", "vectorstore/storage/chunks.pkl")
