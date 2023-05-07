from typing import Optional
from core.conversation.interfaces import ConversationInterface
from lib.agent_log import agent_log
from lib.language.interfaces import (
    Prompt,
    PromptMessage,
)

from .interfaces import AgentInterface
from .base import BaseAgent


class ResearchAgent(BaseAgent, AgentInterface):
    def _get_initial_prompt_text(self, input: str):
        return f"""     
        You are pretending to be an researcher named {self.name} 
        who is concerned with assessing the truth of the following statement:

        {input}
           
        You must decide whether you think the statement is more likely to be mostly true or mostly false.

        Include citations from your research.

        You may only use information directly found in your research.

        However, ultimately you must decide whether you think the statement is mostly "true" or mostly "false."

        Walk through your thinking step by step, before announcing your conclusion.

        Regardless of how ambigious the situation is you must say what you think is more likely to be true.
        """

    def _get_question_prompt_text(self, input: str):
        return f"""     
        You are pretending to be an researcher named {self.name} 
        who is concerned with assessing the truth of the following statement:

        {input}
           
        The next few message will be what you've learned after extensive research.

        You must decide whether you think the statement is more likely to be true or false.

        To do this you may ask a question.

        Think about what would be useful to know to determine if the statement is true or false.
        
        Express the question in as few words as possible.

        Please respond only with the question you wish to ask.
        """

    def _get_heuristics_prompt_text(self, input: str):
        return f"""     
        You are pretending to be an researcher named {self.name} 
        who is concerned with assessing the truth of the following statement:

        {input}
           
        The next few message will be what you've learned after extensive research.

        You must decide whether you think the statement is more likely to be true or false.

        You should put together a list of heuristics that you can evaluate to determine
        the validity of the statement.
        """

    async def _build_heuristics_prompt(
        self,
        input: str,
        relevant_memories: str,
    ) -> Prompt:
        messages = [
            PromptMessage(
                role="system",
                content=self._get_heuristics_prompt_text(input),
            ),
            PromptMessage(
                role="assistant",
                content=f"""The statement I have been asked to assess is:

                {input}

                In my research I have found:

                {self.memory_store.memories_as_list(relevant_memories)}

                Given that, here is how I think I should go about thinking about the statement
                in order to assess it's validity:
                """,
            ),
        ]

        return Prompt(messages=messages)

    async def _build_completion_prompt(
        self,
        input: str,
        conversation: Optional[ConversationInterface] = None,
    ) -> Prompt:
        relevant_memories = await self.memory_store.recall_recent(
            n_memories=self.n_memories_per_prompt,
        )

        heuristics = (
            await self.client.get_completions(
                [
                    await self._build_heuristics_prompt(
                        input, relevant_memories
                    )
                ],
                **self._hyperparameters,
            )
        )[0]

        agent_log.thought(f"{self.name} thinks: {heuristics}")

        messages = [
            PromptMessage(
                role="system",
                content=self._get_initial_prompt_text(input),
            ),
            PromptMessage(
                role="assistant",
                content=f"""As {self.name}, a research trying to determine if the following
                statement is true or false:

                {input}

                In my research I have found:

                {self.memory_store.memories_as_list(relevant_memories)}

                I have decided to use the following approach to assess the validity of the statement:

                {heuristics} 

                Therefore, I have made a decision about whether the statement is true or false. 
                
                Answer: Let's work this out in a step by step way to make sure we have the right answer.
                """,
            ),
        ]

        return Prompt(messages=messages)
