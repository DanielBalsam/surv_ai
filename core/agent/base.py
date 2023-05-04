from typing import Optional
from colorama import Fore

from random import Random, choice

from abc import ABC, abstractmethod

from core.tools.interfaces import ToolbeltInterface
from lib.language.interfaces import (
    LargeLanguageModelClientInterface,
    Prompt,
)
from core.memory_store.interfaces import MemoryStoreInterface
from core.memory_store.local import LocalMemoryStore
from core.tools.toolbelt import Toolbelt
from core.conversation.interfaces import ConversationInterface
from lib.agent_log import agent_log


class BaseAgent(ABC):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        memory_store: Optional[MemoryStoreInterface] = None,
        toolbelt: Optional[list[ToolbeltInterface]] = None,
        random_state: int = 42,
        n_memories_per_prompt: int = 5,
        _hyperparameters: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        if memory_store is None:
            memory_store = LocalMemoryStore()

        if not toolbelt:
            toolbelt = Toolbelt(client, [])

        if not name:
            name = self.__class__.__name__

        self.client = client
        self.memory_store = memory_store
        self.toolbelt = toolbelt

        self.n_memories_per_prompt = n_memories_per_prompt

        self._hyperparameters = _hyperparameters or {}
        self.name = name

        self._color = choice(
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

        self._random = Random()
        self._random.seed(random_state)

    @abstractmethod
    async def _build_completion_prompt(
        self, input: str, conversation: Optional[ConversationInterface] = None
    ) -> Prompt:
        ...

    async def use_tools(
        self, input: str, conversation: Optional[ConversationInterface] = None
    ) -> tuple[str, str]:
        question = input

        question_prompt = await self._build_questions_prompt(
            input, conversation
        )
        question = (await self.client.get_completions([question_prompt]))[0]

        agent_log.thought(f"{self.name} wonders: {question}")

        learned_knowledge = await self.toolbelt.inspect(question, conversation)

        if learned_knowledge:
            for memory in learned_knowledge:
                agent_log.thought(f"{self.name} learned: {memory.text}")

                await self.memory_store.add_memory(memory)

        return question, learned_knowledge

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

    async def teach(self, input: str, source: Optional[str] = "User"):
        await self.memory_store.add_text(f"{source}: {input}", source=source)

    async def transfer_memories(self, agent: "BaseAgent"):
        agent_log.thought(
            f"Transferring memories from {agent.name} to {self.name}"
        )

        for memory in agent.memory_store.memories:
            await self.memory_store.add_memory(memory)
