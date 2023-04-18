from typing import Optional
from core.tools.interfaces import ToolbeltInterface

from abc import ABC, abstractmethod

from lib.language.interfaces import (
    LargeLanguageModelClientInterface,
    PromptMessage,
    Prompt,
)
from core.memory_store.interfaces import MemoryStoreInterface
from core.memory_store.local import LocalMemoryStore
from core.tools.toolbelt import Toolbelt


class BaseAgent(ABC):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        memory_store: Optional[MemoryStoreInterface] = None,
        toolbelt: Optional[list[ToolbeltInterface]] = None,
        self_memory: bool = True,
        _hyperparameters: Optional[dict] = None,
    ):
        if memory_store is None:
            memory_store = LocalMemoryStore()

        if not toolbelt:
            toolbelt = Toolbelt(client, [])

        self.client = client
        self.memory_store = memory_store
        self.toolbelt = toolbelt

        self.messages = []

        self.context = None

        self.self_memory = self_memory
        self._hyperparameters = _hyperparameters or {}

    @abstractmethod
    async def _build_context_prompt(self, input: str) -> Prompt:
        ...

    @abstractmethod
    async def _build_completion_prompt(self, input: str) -> Prompt:
        ...

    async def prompt(self, input: str) -> str:
        if not self.context:
            context_prompt = await self._build_context_prompt(input)
            self.context = (
                await self.client.get_completions(
                    [context_prompt], **self._hyperparameters
                )
            )[0]

        await self.toolbelt.inspect(
            f"{input}]\n\n context: {self.context}", self.memory_store
        )

        prompt = await self._build_completion_prompt(input)

        response = (
            await self.client.get_completions(
                [prompt], **self._hyperparameters
            )
        )[0]

        self.messages.append(PromptMessage(role="assistant", content=response))

        if self.self_memory:
            await self.memory_store.add_text(
                f"AI Assistant response: {response}", source="AI assistant"
            )

        return response

    async def teach(self, input: str, source: Optional[str] = "user"):
        await self.memory_store.add_text(
            f"User guidance: {input}", source=source
        )

    async def transfer_memories(self, agent: "BaseAgent"):
        for memory in agent.memory_store.memories:
            await self.memory_store.add_memory(memory)
