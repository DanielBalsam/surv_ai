from typing import Protocol

from pydantic import BaseModel

from lib.language.interfaces import LargeLanguageModelClientInterface


class AssemblyResponse(BaseModel):
    in_favor: int
    against: int
    undecided: int
    error: int

    percent_in_favor: float
    uncertainty: float
    summaries: list[list[str]]


class AssemblyInterface(Protocol):
    def __init__(
        self, client: LargeLanguageModelClientInterface, *args, **kwargs
    ):
        ...

    async def prompt(self, input: str) -> AssemblyResponse:
        ...
