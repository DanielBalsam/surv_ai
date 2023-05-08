from typing import Optional

from core.embeddings.interfaces import EmbeddingInterface
from core.embeddings.sbert import SentenceBertEmbedding
from lib.vector.search import VectorSearch, VectorSearchType

from .interfaces import Memory, MemoryStoreInterface


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

    def filter_memories(
        self,
        include_sources: Optional[list[str]] = None,
        exclude_sources: Optional[list[str]] = None,
    ):
        memories = self.memories
        if include_sources:
            memories = [
                memory
                for memory in memories
                if memory.source in include_sources
            ]
        if exclude_sources:
            memories = [
                memory
                for memory in memories
                if memory.source not in exclude_sources
            ]

        return memories

    async def recall_relevant(
        self,
        input: str,
        n_memories=5,
        include_sources: Optional[list[str]] = None,
        exclude_sources: Optional[list[str]] = None,
    ) -> list[Memory]:
        embedding = self.get_embedding(input)

        _, indices = VectorSearch.sort_by_similarity(
            [
                m.embedding
                for m in self.filter_memories(include_sources, exclude_sources)
            ],
            embedding,
            type=VectorSearchType.L2,
        )

        return [self.memories[i] for i in indices[:n_memories]]

    async def recall_recent(
        self,
        n_memories=5,
        include_sources: Optional[list[str]] = None,
        exclude_sources: Optional[list[str]] = None,
    ) -> list[Memory]:
        return [
            memory
            for memory in self.filter_memories(
                include_sources, exclude_sources
            )
        ][-n_memories:]

    @staticmethod
    def memories_as_list(memories: list[Memory]) -> str:
        return (
            "\n".join(f"{i + 1}. {m.text}" for i, m in enumerate(memories))
            if memories
            else ""
        )
