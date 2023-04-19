from typing import Protocol, Optional
from core.tools.interfaces import ToolbeltInterface

from lib.language.interfaces import LargeLanguageModelClientInterface


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

    async def prompt(self, input: str, **kwargs) -> str:
        ...
