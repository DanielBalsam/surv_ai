from lib.language.interfaces import (
    Prompt,
    PromptMessage,
)

from .interfaces import AgentInterface
from .base import BaseAgent


class BinaryAgent(BaseAgent, AgentInterface):
    def _get_initial_prompt_text(self, assertion: str):
        return f"""        
        Your job is to determine if the user believes a statement is true or false.

        Here is the original assertion:
        
        "{assertion}"

        The next message will be the statement for the user.

        Your only options are "True," or "False."
        """

    async def _build_completion_prompt(
        self, input: str, conversation=None
    ) -> str:
        self.messages = [
            PromptMessage(
                role="system",
                content=self._get_initial_prompt_text(input),
            ),
            PromptMessage(
                role="user",
                content=conversation.as_string(),
            ),
        ]

        return Prompt(messages=self.messages)
