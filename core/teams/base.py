from abc import ABC, abstractmethod
from random import Random
from typing import Optional
from core.teams.interfaces import TeamResult

from lib.language.interfaces import LargeLanguageModelClientInterface
from core.tools.interfaces import ToolbeltInterface


class BaseTeam(ABC):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        toolbelt: Optional[list[ToolbeltInterface]] = None,
        rounds: int = 3,
        exchanges_per_round: int = 2,
        random_state: int = 42,
        name: Optional[str] = None,
    ):
        if not name:
            name = self.__class__.__name__

        self.client = client
        self.toolbelt = toolbelt

        self.max_conversation_length = rounds * exchanges_per_round

        self.name = name
        self._random = Random()
        self._random.seed(random_state)

    @abstractmethod
    async def _converse(self, input: str) -> TeamResult:
        ...

    async def prompt(self, input: str) -> TeamResult:
        return await self._converse(input)
