import os
import time

from loaders.factory import LoaderFactory
from processing.document_processor import DocumentProcessor
from processing.text_splitter import TextSplitter
from embeddings.embedding import EmbeddingModel
from vectorstore.faiss_store import FAISSStore
from llm.gemini_chat import GeminiChat
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from utils.logger import get_logger


class RAGService:

    def __init__(self):
        self.logger = get_logger(__name__)
        self.embedding_model = EmbeddingModel()
        self.vector_db = FAISSStore()
        self.chat = GeminiChat()
        self.processor = DocumentProcessor()
        self.splitter = TextSplitter(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        self.current_file = None

    # ----------------------------------------------------
    # Process Uploaded Document
    # ----------------------------------------------------

    def process_document(self, file_path):
        try:
            # Don't process again if same file
            if self.current_file == file_path and self.vector_db.index is not None:
                self.logger.info(f"File already processed: {file_path}")
                return {
                    "status": "success",
                    "chunks": len(self.vector_db.chunks),
                    "message": "Already Processed"
                }

            self.logger.info(f"Processing new document: {file_path}")
            loader = LoaderFactory.get_loader(file_path)
            
            raw_pages = loader.load(file_path)
            self.logger.info(f"Loaded {len(raw_pages)} pages from {file_path}")

            processed_pages = self.processor.process(raw_pages)
            
            chunks = self.splitter.split_text(processed_pages)
            self.logger.info(f"Generated {len(chunks)} chunks")

            t0 = time.time()
            vectors = self.embedding_model.embed_documents(chunks)
            embedding_time = time.time() - t0
            self.logger.info(f"Generated embeddings in {embedding_time:.2f}s")

            self.vector_db.create_index(vectors, chunks)
            self.current_file = file_path

            return {
                "status": "success",
                "chunks": len(chunks),
                "message": "Document Processed Successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {e}")
            return {
                "status": "error",
                "message": f"Failed to process document: {str(e)}"
            }

    # ----------------------------------------------------
    # Ask Question
    # ----------------------------------------------------

    def ask(self, question):
        try:
            if self.vector_db.index is None:
                return {
                    "answer": "Please upload a document first.",
                    "sources": [],
                    "metrics": {}
                }

            total_t0 = time.time()

            # Retrieve
            retrieval_t0 = time.time()
            query_vector = self.embedding_model.embed_text(question)
            results = self.vector_db.search(query_vector, top_k=TOP_K)
            retrieval_time = time.time() - retrieval_t0

            self.logger.info(f"Retrieved {len(results)} chunks for question in {retrieval_time:.2f}s")

            # Build context for Gemini with citations
            context_parts = []
            for result in results:
                page_str = f"Page {result['page_start']}"
                if result['page_start'] != result['page_end']:
                    page_str += f"-{result['page_end']}"
                context_parts.append(f"[{page_str} | Chunk {result['chunk_id']}]\n{result['text']}")
            
            context = "\n\n".join(context_parts)

            # Generate answer
            gen_t0 = time.time()
            answer = self.chat.generate_answer(
                question=question,
                context=context
            )
            generation_time = time.time() - gen_t0

            total_response_time = time.time() - total_t0
            self.logger.info(f"Generated answer in {generation_time:.2f}s (Total: {total_response_time:.2f}s)")

            # Return answer, sources and metrics
            return {
                "answer": answer,
                "sources": results,
                "metrics": {
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": total_response_time,
                    "chunks_retrieved": len(results)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            return {
                "answer": "Sorry, I encountered an error while trying to answer your question.",
                "sources": [],
                "metrics": {}
            }