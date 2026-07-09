from pypdf import PdfReader
from loaders.base_loader import BaseLoader


class PDFLoader(BaseLoader):

    def load(self, file_path: str) -> list[dict]:
        try:
            reader = PdfReader(file_path)

            pages = []

            for i, page in enumerate(reader.pages):
                extracted = page.extract_text()

                if extracted:
                    pages.append({
                        "page": i + 1,
                        "text": extracted
                    })

            return pages

        except Exception as e:
            raise Exception(f"Error reading PDF: {e}")