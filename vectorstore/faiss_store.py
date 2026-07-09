import faiss
import numpy as np
import pickle
import os


class FAISSStore:

    def __init__(self):

        self.index = None
        self.chunks = []

    # ----------------------------------------------------
    # Create FAISS Index
    # ----------------------------------------------------

    def create_index(self, vectors, chunks):

        vectors = np.array(vectors).astype("float32")

        dimension = vectors.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(vectors)

        self.chunks = chunks

    # ----------------------------------------------------
    # Search Similar Chunks
    # ----------------------------------------------------

    def search(self, query_vector, top_k=5):

        query_vector = np.array([query_vector]).astype("float32")

        distances, indices = self.index.search(query_vector, top_k)

        results = []

        for score, idx in zip(distances[0], indices[0]):

            if idx == -1:
                continue

            chunk = self.chunks[idx]
            
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "page_start": chunk.get("page_start", 1),
                    "page_end": chunk.get("page_end", 1),
                    "score": float(score)
                }
            )

        return results

    # ----------------------------------------------------
    # Save Index
    # ----------------------------------------------------

    def save_index(self):

        os.makedirs("vectorstore/storage", exist_ok=True)

        faiss.write_index(
            self.index,
            "vectorstore/storage/index.faiss"
        )

        with open(
            "vectorstore/storage/chunks.pkl",
            "wb"
        ) as file:

            pickle.dump(self.chunks, file)

    # ----------------------------------------------------
    # Load Index
    # ----------------------------------------------------

    def load_index(self):

        self.index = faiss.read_index(
            "vectorstore/storage/index.faiss"
        )

        with open(
            "vectorstore/storage/chunks.pkl",
            "rb"
        ) as file:

            self.chunks = pickle.load(file)

    # ----------------------------------------------------
    # Check if Index Exists
    # ----------------------------------------------------

    def index_exists(self):

        return (
            os.path.exists("vectorstore/storage/index.faiss")
            and
            os.path.exists("vectorstore/storage/chunks.pkl")
        )