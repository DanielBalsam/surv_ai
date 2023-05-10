from core.conversation.interfaces import ConversationInterface
from lib.llm.interfaces import (
    Prompt,
    PromptMessage,
)

from ...interfaces import AgentInterface
from ...agent import BaseAgent

from lib.log import logger


class DebateModeratorAgent(BaseAgent, AgentInterface):
    def _get_rubric_prompt_text(self, debate_topic: str):
        return f"""     
        You are pretending to be a team lead named {self.name}.
         
        You need to make a decision for your team of analysts.
        
        You are currently watching a debate between two of your reports (Hank and Sabrina) about an important decision.

        The topic of the debate is: "{debate_topic}"

        The next message will be the last few exchanges from the debate.

        Your should come up with a rubric that you can use to determine who won.
        """

    def _get_grading_prompt_text(self, debate_topic: str):
        return f"""     
        You are pretending to be a team lead named {self.name}.
         
        You need to make a decision for your team of analysts.
        
        You are currently watching a debate between two of your reports (Hank and Sabrina) about an important decision.

        The topic of the debate is: "{debate_topic}"

        The next few messages will be the last few exchanges from the debate.

        You have created a rubric to assess who the winner is.

        Please grade both analysts using the rubric.

        Please always refer to the analysts by their names: Hank and Sabrina.
        """

    def _get_initial_prompt_text(self, debate_topic: str):
        return f"""     
        You are pretending to be a team lead named {self.name}.
         
        You need to make a decision for your team of analysts.
        
        You are currently watching a debate between two of your reports (Hank and Sabrina) about an important decision.

        The topic of the debate is: "{debate_topic}"

        The next few messages will be the last few exchanges from the debate.

        Please summarize the debate so far, and who you you are more persuaded by.

        Include as many specific details each analyst has said to support their argument as possible.

        Include any citations the analysts have mentioned.

        You must let them know who you believe has the more convincing argument, and explain why.

        Please always refer to the analysts by their names: Hank and Sabrina.
        """

    def _get_question_prompt_text(self, debate_topic: str):
        return f"""     
        You are pretending to be a team lead named {self.name}.
         
        You need to make a decision for your team of analysts.
        
        You are currently watching a debate between two of your reports (Hank and Sabrina) about an important decision.

        The next few messages will be the last few exchanges from the debate.

        The topic of the debate is: "{debate_topic}"

        You must think of a question that will allow you to assess who is right in this debate.

        Please respond with only the question that will help you figure out who is correct.
        """

    async def _build_questions_prompt(
        self, debate_topic: str, conversation=None
    ):
        return Prompt(
            messages=[
                PromptMessage(
                    role="system",
                    content=self._get_question_prompt_text(debate_topic),
                ),
                *[
                    PromptMessage(
                        role="assistant"
                        if message.speaker == self.name
                        else "user",
                        content=message.text,
                        name=self.name.replace(" ", "_"),
                    )
                    for message in conversation
                ],
                PromptMessage(
                    role="assistant",
                    content="As a team lead, this is the question I will ask to help decide who I think is correct:",
                ),
            ]
        )

    def _get_rubric_prompt(
        self, debate_topic: str, conversation: ConversationInterface
    ):
        return Prompt(
            messages=[
                PromptMessage(
                    role="system",
                    content=self._get_rubric_prompt_text(debate_topic),
                ),
                *[
                    PromptMessage(
                        role="assistant"
                        if message.speaker == self.name
                        else "user",
                        content=message.text,
                        name=self.name.replace(" ", "_"),
                    )
                    for message in conversation
                ],
                PromptMessage(
                    role="assistant",
                    content="As a team lead, this is the rubric I will use to decide who I think is correct:",
                ),
            ]
        )

    def _get_grading_prompt(
        self,
        debate_topic: str,
        rubric: str,
        relevant_knowledge: str,
        conversation=None,
    ):
        return Prompt(
            messages=[
                PromptMessage(
                    role="system",
                    content=self._get_grading_prompt_text(debate_topic),
                ),
                *[
                    PromptMessage(
                        role="assistant"
                        if message.speaker == self.name
                        else "user",
                        content=message.text,
                        name=self.name.replace(" ", "_"),
                    )
                    for message in conversation
                ],
                PromptMessage(
                    role="assistant",
                    content=f"""I have come up with rubric to determine a winner:
                    
                    {rubric}

                    Here is what I know to be true:

                    {relevant_knowledge}
                    
                    Given this rubric and what I know to be true, here is how I think Hank and Sabrina have done:
                    """,
                ),
            ]
        )

    async def _build_completion_prompt(
        self, conversation: ConversationInterface
    ) -> str:
        relevant_knowledge = self.knowledge_store.recall_recent(
            n_knowledge_items=self.n_knowledge_items_per_prompt,
            exclude_sources=["Debate topic"],
        )
        debate_topic = self.knowledge_store.recall_recent(
            n_knowledge_items=1,
            include_sources=["Debate topic"],
        )[0]

        rubric_prompt = self._get_rubric_prompt(
            debate_topic, conversation=conversation
        )
        rubric = (
            await self.client.get_completions(
                [rubric_prompt], **self._hyperparameters
            )
        )[0]
        logger.log_internal(f"{self.name} plans: {rubric}")

        grading_prompt = self._get_grading_prompt(
            debate_topic, rubric, relevant_knowledge, conversation=conversation
        )
        grades = (
            await self.client.get_completions(
                [grading_prompt], **self._hyperparameters
            )
        )[0]
        logger.log_internal(f"{self.name} plans: {grades}")

        messages = [
            PromptMessage(
                role="system",
                content=self._get_initial_prompt_text(debate_topic),
            ),
            *[
                PromptMessage(
                    role="assistant"
                    if message.speaker == self.name
                    else "user",
                    content=message.text,
                    name=self.name.replace(" ", "_"),
                )
                for message in conversation
            ],
            PromptMessage(
                role="assistant",
                content=f"""I will make my decision using this rubric:
                
                {rubric}

                Here is what I know to be true:

                {relevant_knowledge}

                Given my rubric and what I know to be true, I have assigned these scores:

                {grades}

                Thus, between Hank and Sabrina, the winner is:
                """,
            ),
        ]

        return Prompt(messages=messages)
