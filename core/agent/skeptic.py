from lib.language.interfaces import (
    Prompt,
    PromptMessage,
)

from .interfaces import AgentInterface
from .base import BaseAgent


class SkepticAgent(BaseAgent, AgentInterface):
    def _get_initial_prompt_text(self, memory_list: str):
        return f"""        
        Your job is to ask difficult questions that intellectually challenge a prompt provided to you.

        Domain of questions: {self.context}

        {memory_list}

        The next message will be an assertion from the user.

        You may not agree with the assertion, as you must ask questions that challenge the assertion.

        You should ask for specific examples as often as possible.

        Please only respond with a single question.

        If you do not have any further questions, you can respond with "I have no questions."
        """

    async def _build_completion_prompt(
        self,
        input: str,
    ) -> str:
        relevant_memories = await self.memory_store.recall(
            input + self.context
        )

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

    async def _build_context_prompt(self, input: str) -> str:
        return Prompt(
            messages=[
                PromptMessage(
                    role="system",
                    content=f"""A user will present you with an assertion.

                    Your job will be to categorize this assertion with a few words.

                    For instance, if the assertion is "Paris is the capital of France", then you might respond with "geography".
                    """,
                ),
                PromptMessage(
                    role="user",
                    content=f"""{input}""",
                ),
            ]
        )
