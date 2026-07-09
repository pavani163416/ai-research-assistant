class TextSplitter:
    """
    Splits long text into smaller chunks across page boundaries,
    preserving page_start and page_end metadata.
    """

    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, pages: list[dict]) -> list[dict]:
        # Combine text and build a character mapping to pages
        full_text = ""
        char_to_page = []

        for p in pages:
            text = p["text"]
            page_num = p["page"]
            
            # If not the first page, add a double newline to separate
            if full_text:
                full_text += "\n\n"
                char_to_page.extend([page_num, page_num])

            full_text += text
            char_to_page.extend([page_num] * len(text))

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(full_text):
            end = start + self.chunk_size

            chunk_text = full_text[start:end]
            
            # Determine page range for this chunk
            actual_end = min(end, len(full_text)) - 1
            if actual_end < 0:
                break
                
            page_start = char_to_page[start]
            page_end = char_to_page[actual_end]

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "page_start": page_start,
                "page_end": page_end
            })

            start += self.chunk_size - self.overlap
            chunk_id += 1

        return chunks