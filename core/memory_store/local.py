from typing import Optional

from lib.vector.search import VectorSearch, VectorSearchType
from core.embeddings.interfaces import EmbeddingInterface

from core.embeddings.sbert import SentenceBertEmbedding
from .interfaces import (
    MemoryStoreInterface,
    Memory,
)


class LocalMemoryStore(MemoryStoreInterface):
    def __init__(self, embedding: Optional[EmbeddingInterface] = None):
        if not embedding:
            embedding = SentenceBertEmbedding()

        self.sentence_embeddings = embedding
        self.memories: list[Memory] = []

    def get_embedding(self, text):
        return self.sentence_embeddings.embed([text])[0]

    async def add_text(self, input: str, source: Optional[str] = None):
        embedding = self.get_embedding(input)
        self.memories.append(
            Memory(text=input, embedding=embedding, source=source)
        )

    async def add_memory(self, memory: Memory):
        self.memories.append(memory)

    async def recall(self, input: str, number=5) -> list[Memory]:
        embedding = self.get_embedding(input)

        _, indices = VectorSearch.sort_by_similarity(
            [m.embedding for m in self.memories],
            [embedding],
            type=VectorSearchType.L2,
        )

        return [self.memories[i] for i in indices[:number]]

    @staticmethod
    def memories_as_list(memories: list[Memory]) -> str:
        return (
            "\n".join(f"{i + 1}. {m.text}" for i, m in enumerate(memories))
            if memories
            else ""
        )
