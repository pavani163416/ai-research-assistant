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
            try:
                import streamlit as st
                api_key = st.secrets.get("GEMINI_API_KEY")
            except Exception:
                pass

        if not api_key:
            raise ValueError("Gemini API key not found. Please ensure it's set in .env or Streamlit Secrets.")

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
        import time
        texts = [chunk["text"] for chunk in chunks]
        all_embeddings = []
        batch_size = 90  # Free tier allows 100 requests per minute
        
        import streamlit as st
        progress_text = "Embedding text chunks (this may take a few minutes for large PDFs)..."
        
        # Only show progress bar if we're inside a Streamlit context
        try:
            my_bar = st.progress(0, text=progress_text)
            has_streamlit = True
        except Exception:
            has_streamlit = False

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Embed the batch
            response = genai.embed_content(
                model=self.model_name,
                content=batch,
                task_type="retrieval_document"
            )
            all_embeddings.extend(response['embedding'])
            
            # Update progress
            progress = min((i + len(batch)) / len(texts), 1.0)
            if has_streamlit:
                my_bar.progress(progress, text=f"Processed {len(all_embeddings)}/{len(texts)} chunks. Sleeping to prevent rate limits...")
            else:
                print(f"Processed {len(all_embeddings)}/{len(texts)} chunks...")

            # Sleep to reset the rate limit quota if we have more chunks to process
            if i + batch_size < len(texts):
                time.sleep(62) # Sleep 62 seconds to safely clear the 1-minute window
                
        if has_streamlit:
            my_bar.empty()
            
        return all_embeddings