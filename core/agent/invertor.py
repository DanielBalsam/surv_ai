from lib.language.interfaces import (
    Prompt,
    PromptMessage,
)

from .interfaces import AgentInterface
from .base import BaseAgent


class InvertorAgent(BaseAgent, AgentInterface):
    def _get_initial_prompt_text(self):
        return """
        Your job is to provide a sentence with the exact opposite of an input statement.

        The next message will be your input statement.

        Please provide a single sentence with the opposite sentiment.
        """

    async def _build_completion_prompt(
        self, input: str, conversation=None
    ) -> str:
        self.messages = [
            PromptMessage(
                role="system",
                content=self._get_initial_prompt_text(),
            ),
            PromptMessage(
                role="user",
                content=input,
            ),
            PromptMessage(
                role="assistant",
                content="A sentence with the opposite sentiment would be: ",
            ),
        ]

        return Prompt(messages=self.messages)
