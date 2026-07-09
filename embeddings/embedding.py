from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_text(self, text):

        return self.model.encode(text)

    def embed_documents(self, chunks: list[dict]):

        texts = [chunk["text"] for chunk in chunks]
        return self.model.encode(texts)