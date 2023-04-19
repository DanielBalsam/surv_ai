from lib.language.interfaces import (
    Prompt,
    PromptMessage,
)

from .interfaces import AgentInterface
from .base import BaseAgent


class ModeratorAgent(BaseAgent, AgentInterface):
    def _get_initial_prompt_text(self, memory_list: str):
        return f"""        
        Your job is to determine whether other AI agents are staying on topic to answer the question.

        Domain of questions: {self.context}

        Most recent messages:

        {memory_list}

        The next message will be the original question from the user.

        You may only respond with either "Continue" or "Off Topic".
        """

    async def _build_completion_prompt(
        self,
        input: str,
    ) -> str:
        recent_memories = await self.memory_store.recall_recent()

        self.messages = [
            PromptMessage(
                role="system",
                content=self._get_initial_prompt_text(
                    self.memory_store.memories_as_list(recent_memories),
                ),
            ),
            PromptMessage(role="user", content=input),
        ]

        return Prompt(messages=self.messages)

    async def _build_context_prompt(self, input: str) -> str:
        return Prompt(
            messages=[
                PromptMessage(
                    role="system",
                    content=f"""A user will present you with a question.

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
