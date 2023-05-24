from surv_ai.lib.knowledge_store.interfaces import Knowledge
from surv_ai.lib.llm.interfaces import Prompt, PromptMessage
from surv_ai.lib.log import logger

from ..agent import BaseAgent
from ..interfaces import AgentInterface


class ReasoningAgent(BaseAgent, AgentInterface):
    def _get_completion_prompt_text(self, prompt: str):
        return f"""     
        You are a citizen who must vote on whether this statement is more likely to be true or false:

        {prompt}
           
        You must decide whether you think the statement is more likely to be true or false.

        You must cast your vote and participate in the civic process.

        The next two messages will be arguments in favor and against the above statement.

        You must decide which argument you are more persuaded by. You may not be undecided.
        """

    def _get_argument_prompt_text(self, prompt: str):
        return f"""     
        You are a citizen who must vote on whether this statement is more likely to be true or false:

        {prompt}
           
        You must decide whether you think the statement is more likely to be true or false.

        You must cast your vote and participate in the civic process.

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

    async def _get_plan(self, prompt: str, relevant_knowledge: list[Knowledge]):
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
                content="""In order to determine whether the provided statement is more likely to be true or false,
                This is an outline of how I will think about the problem:
                """,
            ),
        ]

        return (await self.client.get_completions([Prompt(messages=messages)]))[0]

    async def _get_argument_in_favor(
        self,
        prompt: str,
        relevant_knowledge: list[Knowledge],
    ):
        messages = [
            PromptMessage(
                role="system",
                content=self._get_argument_prompt_text(
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
                content=f"""
                Thus after looking at the relevant sources, I have come to the conclusion that the assertion
                "{prompt}" is more likely to be true. I will now explain my reasoning step by step while citing only the above sources:
                """,
            ),
        ]

        return (await self.client.get_completions([Prompt(messages=messages)]))[0]

    async def _get_argument_against(
        self,
        prompt: str,
        relevant_knowledge: list[Knowledge],
    ):
        messages = [
            PromptMessage(
                role="system",
                content=self._get_argument_prompt_text(
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
                content=f"""
                Thus after looking at the relevant sources, I have come to the conclusion that the assertion
                "{prompt}" is more likely to be false. I will now explain my reasoning step by step while citing only the above sources:
                """,
            ),
        ]

        return (await self.client.get_completions([Prompt(messages=messages)]))[0]

    async def _build_completion_prompt(
        self,
        prompt: str,
    ) -> Prompt:
        relevant_knowledge = self.knowledge_store.recall_recent(
            n_knowledge_items=self.n_knowledge_items_per_prompt,
        )

        plan = await self._get_plan(prompt, relevant_knowledge)
        logger.log_internal(f"{self.name} plans: {plan}")

        argument_in_favor = await self._get_argument_in_favor(prompt, relevant_knowledge)
        logger.log_internal(f"{self.name} argues in favor: {argument_in_favor}")

        argument_against = await self._get_argument_against(prompt, relevant_knowledge)
        logger.log_internal(f"{self.name} argues against: {argument_against}")

        messages = [
            PromptMessage(
                role="system",
                content=self._get_completion_prompt_text(
                    prompt,
                ),
            ),
            PromptMessage(
                role="assistant",
                content=argument_in_favor,
                name="Argument In Favor",
            ),
            PromptMessage(
                role="assistant",
                content=argument_against,
                name="Argument Against",
            ),
            PromptMessage(
                role="assistant",
                content=f"""My approach to answering the question will be:

                {plan}
                
                Now, after consider both perspectives I am more persuaded by the argument stating:
                """,
            ),
        ]

        return Prompt(messages=messages)

    async def prompt(self, statement: str, *args, **kwargs) -> str:
        prompt = await self._build_completion_prompt(statement, *args, **kwargs)

        response = (await self.client.get_completions([prompt], **self._hyperparameters))[0]

        return response
