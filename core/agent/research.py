from typing import Optional
from core.tools.interfaces import ToolbeltInterface

from lib.language.interfaces import (
    LargeLanguageModelClientInterface,
    Prompt,
    PromptMessage,
)
from core.memory_store.interfaces import MemoryStoreInterface, Memory
from core.memory_store.local import LocalMemoryStore
from core.tools.toolbelt import Toolbelt

from .interfaces import AgentInterface


class ResearchAgent(AgentInterface):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        memory_store: Optional[MemoryStoreInterface] = None,
        toolbelt: Optional[list[ToolbeltInterface]] = None,
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

    def _get_initial_prompt_text(self, memory_list: str):
        return f"""        
        Your job is to see relevant information to find an answer to subsequent prompt.

        Domain of question: {self.context}

        {memory_list}

        The next message will be a question from the user.

        Please explain your thinking, but never respond with more than a few sentences.  
        """

    def _build_prompt(
        self,
        input: str,
        relevant_memories: list[Memory],
    ) -> str:
        self.messages = [
            PromptMessage(
                role="system",
                content=self._get_initial_prompt_text(
                    self.memory_store.memories_as_list(relevant_memories),
                ),
            ),
            PromptMessage(role="user", content=input),
        ]

        return Prompt(messages=self.messages)

    def build_context_prompt(self, input: str) -> str:
        return Prompt(
            messages=[
                PromptMessage(
                    role="system",
                    content=f"""A user will ask you a question.

                    Your job will be to categorize this question with a few words.

                    For instance, if the question is "What is the capital of France?", then you might respond with "geography".
                    """,
                ),
                PromptMessage(
                    role="user",
                    content=f"""{input}""",
                ),
            ]
        )

    async def prompt(self, input: str) -> str:
        if not self.context:
            context_prompt = self.build_context_prompt(input)
            self.context = (
                await self.client.get_completions([context_prompt])
            )[0]

        await self.toolbelt.inspect(
            f"{input}]\n\n context: {self.context}", self.memory_store
        )

        relevant_memories = await self.memory_store.recall(
            input + self.context
        )

        prompt = self._build_prompt(input, relevant_memories)

        response = (await self.client.get_completions([prompt]))[0]

        self.messages.append(PromptMessage(role="assistant", content=response))

        await self.memory_store.add_text(
            f"AI Assistant response: {response}", source="AI assistant"
        )

        return response

    async def teach(self, input: str, source: Optional[str] = "user"):
        await self.memory_store.add_text(
            f"User guidance: {input}", source=source
        )
