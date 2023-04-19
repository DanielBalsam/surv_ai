from lib.language.interfaces import (
    Prompt,
    PromptMessage,
)

from .interfaces import AgentInterface
from .base import BaseAgent


class ResearchAgent(BaseAgent, AgentInterface):
    def _get_initial_prompt_text(self, memory_list: str):
        return f"""        
        Your job is to see relevant information to find an answer to subsequent prompt.

        Domain of question: {self.context}

        {memory_list}

        The next message will be a question from the user.

        Please explain your thinking, but never respond with more than a few sentences.  
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
