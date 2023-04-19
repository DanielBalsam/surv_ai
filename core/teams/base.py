from abc import ABC, abstractmethod
from typing import Optional

from lib.language.interfaces import LargeLanguageModelClientInterface
from core.tools.interfaces import ToolbeltInterface


class BaseTeam(ABC):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        toolbelt: Optional[list[ToolbeltInterface]] = None,
        n_conversations: int = 3,
        max_conversation_length: int = 10,
    ):
        self.client = client
        self.toolbelt = toolbelt

        self.n_conversations = n_conversations
        self.max_conversation_length = max_conversation_length

    @abstractmethod
    async def _converse(self, input: str) -> str:
        ...

    async def prompt(self, input: str) -> str:
        return await self._converse(input)
