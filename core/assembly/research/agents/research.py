from core.knowledge_store.interfaces import Knowledge
from lib.llm.interfaces import Prompt, PromptMessage
from lib.log import logger

from ...agent import BaseAgent
from ...interfaces import AgentInterface


class ResearchAgent(BaseAgent, AgentInterface):
    def _get_initial_prompt_text(self, prompt: str):
        return f"""     
        You are a decisive agent who is concerned with whether this statement is more likely to be true or false:

        {prompt}
           
        You must decide whether you think the statement is more likely to be true or false.

        You must make a decision. You cannot be undecided under any circumstances.

        The next set of messages will be a series of articles that you can use to help you make your decision.

        In your response please include as many citations from your research as possible.

        Include both the publication title and the article title in your citations.

        You must make the best decision possible with the information you have. 
        """

    def _get_plan_prompt_text(self, prompt: str):
        return f"""     
        You are a decisive agent who is concerned with whether this statement is more likely to be true or false:

        {prompt}
           
        You must decide whether you think the statement is more likely to be true or false.

        The next set of messages will be a series of articles that you can use to help you make your decision.

        In your response please construct a plan for how you will approach the problem.

        Make sure your plan includes offering specific citations.

        You must make the best decision possible with the information you have. You cannot be undecided.
        """

    async def _get_plan(
        self, prompt: str, relevant_knowledge: list[Knowledge]
    ):
        messages = [
            PromptMessage(
                role="system",
                content=self._get_plan_prompt_text(
                    prompt,
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
                content=f"""In order to determine whether the provided statement is more likely to be true or false,
                This is an outline of how I will think about the problem:
                """,
            ),
        ]

        return (
            await self.client.get_completions([Prompt(messages=messages)])
        )[0]

    async def _build_completion_prompt(
        self,
        prompt: str,
    ) -> Prompt:
        relevant_knowledge = self.knowledge_store.recall_recent(
            n_knowledge_items=self.n_knowledge_items_per_prompt,
        )

        plan = await self._get_plan(prompt, relevant_knowledge)
        logger.log_internal(f"{self.name} plans: {plan}")

        messages = [
            PromptMessage(
                role="system",
                content=self._get_initial_prompt_text(
                    prompt,
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
                content=f"""In order to determine whether the provided statement is more likely to be true or false,
                This is an outline of how I will think about the problem:

                ```
                {plan}
                ```

                Thus after looking at the relevant sources, I believe the statement is more likely to be:
                """,
            ),
        ]

        return Prompt(messages=messages)
