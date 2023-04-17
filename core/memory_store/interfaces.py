from typing import Any, Optional, Protocol
from lib.language.interfaces import LargeLanguageModelClientInterface
from pydantic import BaseModel, Field


class Memory(BaseModel):
    text: str
    embedding: Any = Field(repr=False)

    source: Optional[str]


class MemoryStoreInterface(Protocol):
    def __init__(
        self, client: LargeLanguageModelClientInterface, *args, **kwargs
    ):
        ...

    async def recall(self, input: str, number=5, **kwargs) -> list[Memory]:
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
