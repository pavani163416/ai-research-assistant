from loaders.base_loader import BaseLoader


class TXTLoader(BaseLoader):

    def load(self, file_path: str) -> list[dict]:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return [{"page": 1, "text": file.read()}]

        except Exception as e:
            raise Exception(f"Error reading TXT: {e}")