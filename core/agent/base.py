from abc import ABC, abstractmethod
from random import choice
from typing import Optional

from colorama import Fore

from core.conversation.interfaces import ConversationInterface
from core.knowledge_store.interfaces import Knowledge, KnowledgeStoreInterface
from core.knowledge_store.local import LocalKnowledgeStore
from lib.log import logger
from lib.llm.interfaces import LargeLanguageModelClientInterface, Prompt


class BaseAgent(ABC):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        knowledge_store: Optional[KnowledgeStoreInterface] = None,
        n_knowledge_items_per_prompt: int = 5,
        name: Optional[str] = None,
        _hyperparameters: Optional[dict] = None,
    ):
        if knowledge_store is None:
            knowledge_store = LocalKnowledgeStore()

        if not name:
            name = self.__class__.__name__

        self.client = client
        self.knowledge_store = knowledge_store

        self.n_knowledge_items_per_prompt = n_knowledge_items_per_prompt

        self._hyperparameters = _hyperparameters or {}
        self.name = name

        self.color = choice(
            [
                Fore.BLUE,
                Fore.CYAN,
                Fore.GREEN,
                Fore.RED,
                Fore.MAGENTA,
                Fore.YELLOW,
                Fore.LIGHTGREEN_EX,
            ]
        )

    @abstractmethod
    async def _build_completion_prompt(
        self, input: str, conversation: Optional[ConversationInterface] = None
    ) -> Prompt:
        ...

    async def prompt(
        self, input: str, conversation: Optional[ConversationInterface] = None
    ) -> str:
        prompt = await self._build_completion_prompt(input, conversation)

        response = (
            await self.client.get_completions(
                [prompt], **self._hyperparameters
            )
        )[0]

        return response

    def teach_text(self, input: str, source: Optional[str] = "User"):
        self.knowledge_store.add_text(f"{source}: {input}", source=source)

    def teach_knowledge(self, knowledge: Knowledge):
        self.knowledge_store.add_knowledge(knowledge)
