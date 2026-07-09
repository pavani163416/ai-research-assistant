import time
from services.rag_service import RAGService
from utils.config import UPLOAD_FOLDER

def main():

    file_path = f"{UPLOAD_FOLDER}/sample.pdf"

    rag = RAGService()

    # ---------------------------------------------------------
    # Check if FAISS Index Already Exists / Create Index
    # ---------------------------------------------------------

    print("\nProcessing Document...")
    result = rag.process_document(file_path)

    if result["status"] == "error":
        print(f"Error: {result['message']}")
        return

    print(f"Status: {result['message']}")
    print(f"Total Chunks: {result['chunks']}")

    # ---------------------------------------------------------
    # Gemini Chat
    # ---------------------------------------------------------

    print("\n========================================")
    print("AI Research Assistant Ready")
    print("Type 'exit' to quit.")
    print("========================================\n")

    while True:

        question = input("Ask a Question : ")

        if question.lower() == "exit":
            print("\nGoodbye!\n")
            break

        print("\nThinking...")
        response = rag.ask(question)

        answer = response.get("answer", "Error")
        sources = response.get("sources", [])
        metrics = response.get("metrics", {})

        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)
        print("-" * 80)
        
        if metrics:
            print(f"Metrics: Retrieval {metrics.get('retrieval_time',0):.2f}s | "
                  f"Generation {metrics.get('generation_time',0):.2f}s | "
                  f"Total {metrics.get('total_time',0):.2f}s")
        
        if sources:
            print("\nSOURCES:")
            for s in sources:
                page_str = f"Page {s['page_start']}"
                if s['page_start'] != s['page_end']:
                    page_str += f"-{s['page_end']}"
                print(f"[{page_str} | Chunk {s['chunk_id']} | Score {s['score']:.4f}]")

        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()