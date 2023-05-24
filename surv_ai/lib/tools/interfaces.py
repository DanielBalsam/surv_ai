from re import Pattern
from typing import Protocol

from ..llm.interfaces import LargeLanguageModelClientInterface


class NoMemoriesFoundException(Exception):
    ...


class ToolInterface(Protocol):
    instruction: str
    command: Pattern

    def __init__(self, _: LargeLanguageModelClientInterface, *args, **kwargs):
        ...

    async def use(self, client: LargeLanguageModelClientInterface, *args, **kwargs):
        ...


class ToolBeltInterface(Protocol):
    tools: list[ToolInterface]

    def __init__(
        self,
        _: LargeLanguageModelClientInterface,
        tools: list[ToolInterface],
    ):
        ...

    async def inspect(self, client: LargeLanguageModelClientInterface, query: str):
        ...
