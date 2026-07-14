import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

from utils.config import EMBEDDING_MODEL

class EmbeddingModel:
    def __init__(self):
        # Initialize local HuggingFace embedding model
        # all-MiniLM-L6-v2 is fast, small (90MB), and good for semantic search
        self.model_name = EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text):
        # sentence_transformers returns a numpy array, we convert to list
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_documents(self, chunks: list[dict]):
        texts = [chunk["text"] for chunk in chunks]
        
        import streamlit as st
        progress_text = f"Embedding {len(texts)} chunks locally (super fast!)..."
        
        try:
            my_bar = st.progress(0, text=progress_text)
            has_streamlit = True
        except Exception:
            has_streamlit = False

        # Encode locally in batches for efficiency (no rate limits!)
        # SentenceTransformers handles batching automatically when passed a list
        embeddings = self.model.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=False
        )
        
        if has_streamlit:
            my_bar.progress(1.0, text="Embeddings generated!")
            my_bar.empty()
            
        return embeddings.tolist()