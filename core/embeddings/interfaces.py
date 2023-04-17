from typing import Protocol


class EmbeddingInterface(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...
