from typing import Optional, Protocol

from pydantic import BaseModel

from core.tools.interfaces import ToolbeltInterface
from lib.language.interfaces import LargeLanguageModelClientInterface


class TeamResult(BaseModel):
    points_in_favor: int
    points_against: int
    points_undecided: int
    summary: list[str]


class TeamInterface(Protocol):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        toolbelt: Optional[list[ToolbeltInterface]] = None,
        n_conversations: int = 3,
        max_conversation_length: int = 10,
        *args,
        **kwargs
    ):
        ...

    async def prompt(self, input: str, **kwargs) -> TeamResult:
        ...
