import re

class DocumentProcessor:
    """
    Dedicated layer for cleaning document text while preserving page metadata.
    """

    def process(self, pages: list[dict]) -> list[dict]:
        processed_pages = []

        for page_data in pages:
            text = page_data["text"]

            # 1. Normalize whitespace (replace multiple spaces with one)
            text = re.sub(r'[ \t]+', ' ', text)

            # 2. Normalize line breaks (replace multiple newlines with at most two)
            text = re.sub(r'\n{3,}', '\n\n', text)

            # 3. Strip leading and trailing whitespace
            text = text.strip()

            # 4. Skip empty pages
            if not text:
                continue

            processed_pages.append({
                "page": page_data["page"],
                "text": text
            })

        return processed_pages
