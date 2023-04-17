from typing import Protocol, Optional

from core.memory_store.interfaces import MemoryStoreInterface
from core.tools.interfaces import ToolbeltInterface
from lib.language.interfaces import LargeLanguageModelClientInterface


class AgentInterface(Protocol):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        memory_store: Optional[MemoryStoreInterface],
        toolbelt: Optional[list[ToolbeltInterface]],
    ):
        ...

    async def prompt(self, input: str) -> str:
        ...

    async def teach(self, input: str):
        ...
