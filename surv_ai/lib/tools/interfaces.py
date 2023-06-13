from re import Pattern
from typing import Protocol

from pydantic import BaseModel

from ..llm.interfaces import LargeLanguageModelClientInterface


class NoMemoriesFoundException(Exception):
    ...


class ToolResult(BaseModel):
    url: str
    site_name: str
    title: str
    body: str


class ToolInterface(Protocol):
    instruction: str
    command: Pattern

    def __init__(self, _: LargeLanguageModelClientInterface, *args, **kwargs):
        ...

    async def use(self, *args, **kwargs) -> list[ToolResult]:
        ...


class ToolBeltInterface(Protocol):
    tools: list[ToolInterface]

    def __init__(
        self,
        _: LargeLanguageModelClientInterface,
        tools: list[ToolInterface],
    ):
        ...

    async def inspect(self, client: LargeLanguageModelClientInterface, query: str) -> list[ToolResult]:
        ...
