from typing import Optional

from core.conversation.interfaces import ConversationInterface
from lib.language.interfaces import Prompt, PromptMessage

from .base import BaseAgent
from .interfaces import AgentInterface


class ResearchAgent(BaseAgent, AgentInterface):
    def _get_initial_prompt_text(self, input: str, relevant_knowledge: str):
        return f"""     
        You are a researcher named who is concerned with determining the truth of the following statement:

        {input}
           
        You must decide whether you think the statement is more likely to be mostly true or mostly false.

        The next set of messages will be a series of articles that you can use to help you make your decision.

        In your response please include as many citations from your research as possible.

        Include both the publication title and the article title in your citation.

        You must end your output simply with "Decision: True" or "Decision: False"
        """

    async def _build_completion_prompt(
        self,
        input: str,
        conversation: Optional[ConversationInterface] = None,
    ) -> Prompt:
        relevant_knowledge = self.knowledge_store.recall_recent(
            n_knowledge_items=self.n_knowledge_items_per_prompt,
        )

        messages = [
            PromptMessage(
                role="system",
                content=self._get_initial_prompt_text(
                    input,
                    self.knowledge_store.knowledge_as_string(
                        relevant_knowledge
                    ),
                ),
            ),
            *[
                PromptMessage(
                    content=knowledge.text,
                    role="user",
                    name=knowledge.source,
                )
                for knowledge in relevant_knowledge
            ],
            PromptMessage(
                role="assistant",
                content=f"""Question: is the following statement more likely to be true or false?

                {input}

                Answer: In order to determine whether the provided statement is more likely to be true or false,
                let's work this out in a step by step way to make sure we have the right answer.
                """,
            ),
        ]

        return Prompt(messages=messages)
