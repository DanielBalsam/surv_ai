from typing import Protocol
from re import Pattern

from lib.language.interfaces import LargeLanguageModelClientInterface


class ToolInterface(Protocol):
    instruction: str
    command: Pattern

    def __init__(
        self, client: LargeLanguageModelClientInterface, *args, **kwargs
    ):
        ...

    async def use(self, *args, **kwargs):
        ...


class ToolbeltInterface(Protocol):
    tools: list[ToolInterface]

    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        tools: list[ToolInterface],
    ):
        ...

    async def observe(self, query: str):
        ...
