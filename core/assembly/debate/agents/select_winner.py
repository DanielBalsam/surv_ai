from lib.llm.interfaces import (
    Prompt,
    PromptMessage,
)

from ...interfaces import AgentInterface
from ...agent import BaseAgent


class SelectWinnerAgent(BaseAgent, AgentInterface):
    def _get_initial_prompt_text(self):
        return f"""        
        Your job is to determine who a team lead named Dan believes is winning a debate between two researchers named "Sabrina" and "Hank."

        The next message will be Dan's summary of the conversation. 

        You may only respond with the name of the researcher that Dan believes is winning.

        Your only options are the names of the debating researchers: Hank, Sabrina or "Undecided."
        """

    async def _build_completion_prompt(self, prompt: str) -> str:
        self.messages = [
            PromptMessage(
                role="system",
                content=self._get_initial_prompt_text(),
            ),
            PromptMessage(
                role="user",
                content=prompt,
            ),
        ]

        return Prompt(messages=self.messages)
