from lib.language.interfaces import (
    Prompt,
    PromptMessage,
)

from .interfaces import AgentInterface
from .base import BaseAgent


class DecisionAgent(BaseAgent, AgentInterface):
    def _get_initial_prompt_text(self, memory_list: str):
        return f"""        
        Your job is to cast a reasonable vote based on the information provided to you.

        Domain of questions: {self.context}

        {memory_list}

        The next message will be a question from the user.

        You must make a decision based on the information provided to you.

        You may not respond with "I don't know" or "I don't understand".

        You may only respond with a single word, or proper noun.
        """

    async def _build_completion_prompt(
        self,
        input: str,
    ) -> str:
        relevant_memories = await self.memory_store.recall_relevant(
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
