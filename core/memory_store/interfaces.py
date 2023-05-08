from typing import Optional, Protocol

from pydantic import BaseModel

from lib.language.interfaces import LargeLanguageModelClientInterface


class Memory(BaseModel):
    text: str
    source: Optional[str]


class MemoryStoreInterface(Protocol):
    def __init__(
        self, client: LargeLanguageModelClientInterface, *args, **kwargs
    ):
        ...

    async def recall_relevant(
        self, input: str, n_memories=5, **kwargs
    ) -> list[Memory]:
        ...

    async def recall_recent(self, n_memories=5, **kwargs) -> list[Memory]:
        ...

    async def add_text(
        self, input: str, source: Optional[str] = None, **kwargs
    ):
        ...

    async def add_memory(self, memory: Memory, **kwargs):
        ...

    @staticmethod
    def memories_as_list(memories: list[Memory]) -> str:
        ...
