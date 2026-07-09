import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

from utils.config import EMBEDDING_MODEL

# Load .env from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class EmbeddingModel:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found.")
        genai.configure(api_key=api_key)
        self.model_name = EMBEDDING_MODEL

    def embed_text(self, text):
        response = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query"
        )
        return response['embedding']

    def embed_documents(self, chunks: list[dict]):
        texts = [chunk["text"] for chunk in chunks]
        response = genai.embed_content(
            model=self.model_name,
            content=texts,
            task_type="retrieval_document"
        )
        return response['embedding']