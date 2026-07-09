from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """
    Base class for all document loaders.
    Every loader must implement load().
    """

    @abstractmethod
    def load(self, file_path: str) -> list[dict]:
        pass