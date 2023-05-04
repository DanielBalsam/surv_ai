from lib.language.interfaces import (
    Prompt,
    PromptMessage,
)

from ..interfaces import AgentInterface
from ..base import BaseAgent


class SelectWinnerAgent(BaseAgent, AgentInterface):
    def _get_initial_prompt_text(self):
        return f"""        
        Your job is to determine who a team lead named Dan believes is winning a debate between two researchers named "Sabrina" and "Hank."

        The next message will be Dan's summary of the conversation. He is never undecided.

        You may only respond with the name of the researcher that Dan believes is winning.

        Your only options are the names of the debating researchers: Hank or Sabrina
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
        ]

        return Prompt(messages=self.messages)
