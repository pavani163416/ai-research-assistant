from docx import Document
from loaders.base_loader import BaseLoader


class DOCXLoader(BaseLoader):

    def load(self, file_path: str) -> list[dict]:
        try:
            document = Document(file_path)

            text = "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

            return [{"page": 1, "text": text}]

        except Exception as e:
            raise Exception(f"Error reading DOCX: {e}")