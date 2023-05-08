from lib.language.interfaces import (
    Prompt,
    PromptMessage,
)

from ..interfaces import AgentInterface
from ..base import BaseAgent

from lib.agent_log import agent_log


class TeamLeadAgent(BaseAgent, AgentInterface):
    def _get_rubric_prompt_text(self, input: str):
        return f"""     
        You are pretending to be a team lead named {self.name}.
         
        You need to make a decision for your team of analysts.
        
        You are currently watching a debate between two of your reports (Hank and Sabrina) about an important decision.

        The topic of the debate is: "{input}"

        The next message will be the last few exchanges from the debate.

        Your should come up with a rubric that you can use to determine who won.
        """

    def _get_grading_prompt_text(self, input: str):
        return f"""     
        You are pretending to be a team lead named {self.name}.
         
        You need to make a decision for your team of analysts.
        
        You are currently watching a debate between two of your reports (Hank and Sabrina) about an important decision.

        The topic of the debate is: "{input}"

        The next few messages will be the last few exchanges from the debate.

        You have created a rubric to assess who the winner is.

        Please grade both analysts using the rubric.

        Please always refer to the analysts by their names: Hank and Sabrina.
        """

    def _get_initial_prompt_text(self, input: str):
        return f"""     
        You are pretending to be a team lead named {self.name}.
         
        You need to make a decision for your team of analysts.
        
        You are currently watching a debate between two of your reports (Hank and Sabrina) about an important decision.

        The topic of the debate is: "{input}"

        The next few messages will be the last few exchanges from the debate.

        Please summarize the debate so far, and who you you are more persuaded by.

        Include as many specific details each analyst has said to support their argument as possible.

        Include any citations the analysts have mentioned.

        You must let them know who you believe has the more convincing argument, and explain why.

        Please always refer to the analysts by their names: Hank and Sabrina.
        """

    def _get_question_prompt_text(self, input: str):
        return f"""     
        You are pretending to be a team lead named {self.name}.
         
        You need to make a decision for your team of analysts.
        
        You are currently watching a debate between two of your reports (Hank and Sabrina) about an important decision.

        The next few messages will be the last few exchanges from the debate.

        The topic of the debate is: "{input}"

        You must think of a question that will allow you to assess who is right in this debate.

        Please respond with only the question that will help you figure out who is correct.
        """

    def _get_reflection_prompt_text(self, input: str):
        return f"""     
        You are pretending to be a team lead named {self.name}.
         
        You need to make a decision for your team of analysts.
        
        You are currently watching a debate between two of your reports (Hank and Sabrina) about an important decision.

        The topic of the debate is: "{input}"

        The next few messages will be the last few exchanges from the debate.

        The following message will be your first draft at a decision.

        Please reflect and revise this draft.

        Please fix any logical errors, and make sure that your argument is as strong as possible.

        Please always refer to the analysts by their names: Hank and Sabrina.

        Respond with your revised draft.
        """

    async def _build_questions_prompt(self, input: str, conversation=None):
        return Prompt(
            messages=[
                PromptMessage(
                    role="system",
                    content=self._get_question_prompt_text(input),
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

    def _get_rubric_prompt(self, input: str, conversation=None):
        return Prompt(
            messages=[
                PromptMessage(
                    role="system",
                    content=self._get_rubric_prompt_text(input),
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
        input: str,
        rubric: str,
        relevant_memories: str,
        conversation=None,
    ):
        return Prompt(
            messages=[
                PromptMessage(
                    role="system",
                    content=self._get_grading_prompt_text(input),
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

                    {relevant_memories}
                    
                    Given this rubric and what I know to be true, here is how I think Hank and Sabrina have done:
                    """,
                ),
            ]
        )

    async def _build_completion_prompt(
        self, input: str, conversation=None
    ) -> str:
        relevant_memories = await self.memory_store.recall_recent(
            n_memories=self.n_memories_per_prompt,
        )

        rubric_prompt = self._get_rubric_prompt(
            input, conversation=conversation
        )
        rubric = (
            await self.client.get_completions(
                [rubric_prompt], **self._hyperparameters
            )
        )[0]
        agent_log.thought(f"{self.name} thinks: {rubric}")

        grading_prompt = self._get_grading_prompt(
            input, rubric, relevant_memories, conversation=conversation
        )
        grades = (
            await self.client.get_completions(
                [grading_prompt], **self._hyperparameters
            )
        )[0]
        agent_log.thought(f"{self.name} thinks: {grades}")

        messages = [
            PromptMessage(
                role="system",
                content=self._get_initial_prompt_text(input),
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

                {relevant_memories}

                Given my rubric and what I know to be true, I have assigned these scores:

                {grades}

                Thus, between Hank and Sabrina, the winner is:
                """,
            ),
        ]

        return Prompt(messages=messages)
