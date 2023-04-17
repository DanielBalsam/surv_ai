from typing import Protocol
from pydantic import BaseModel

from lib.language.interfaces import LargeLanguageModelClientInterface


class AssemblyResponse(BaseModel):
    final_response: str
    percent_in_favor: float

    dissenting_responses: list[str]


class AssemblyInterface(Protocol):
    def __init__(
        self, client: LargeLanguageModelClientInterface, *args, **kwargs
    ):
        ...

    async def ask(self, input: str) -> AssemblyResponse:
        ...
