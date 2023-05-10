from core.conversation.interfaces import ConversationInterface
from lib.log import logger
from lib.llm.interfaces import (
    Prompt,
    PromptMessage,
)

from ...interfaces import AgentInterface
from ...agent import BaseAgent


class DebaterAgent(BaseAgent, AgentInterface):
    def _get_initial_prompt_text(self, beliefs: str):
        return f"""     
        You are pretending to be an analyst named {self.name} who has a strong belief about a topic.

        Through intensive analysis you have concluded the following with very strong conviction:

        {beliefs}
           
        Your job is to seek relevant information to support your belief and field answers to further question.

        The next message will be the lastest few statements from a debate, including your own.

        You should respond directly to your opponent's points.

        Include citation from your research.

        Use strong rhetoric, but concrete examples.

        Be as nuanced and thoughtful as possible.
        """

    def _get_question_prompt_text(self, beliefs: str):
        return f"""     
        You are pretending to be an analyst named {self.name} who has a strong belief about a topic.

        Through intensive analysis you have concluded the following with very strong conviction:

        {beliefs}
           
        Your job is to seek relevant information to support your belief and field answers to further question.

        The next message will be the lastest few statements from a debate, including your own.

        You should then respond with a up to three questions to help guide your research.

        Think about what would be useful to know to help win the debate.

        Think about what question will help bring strong evidence to your argument.

        Please respond only with the questions you wish to ask.
        """

    def _get_reflection_prompt(self, beliefs: str):
        return f"""     
        You are pretending to be an analyst named {self.name} who has a strong belief about a topic.

        Through intensive analysis you have concluded the following with very strong conviction:

        {beliefs}
           
        Your job is to seek relevant information to support your belief and field answers to further question.

        Do not use any information from outside of your analysis.

        The next message will be the lastest few statements from a debate, including your own.

        Then you will receive a first draft of your response. 
        
        Please reflect and revise this draft.

        Fix any logical errors, and make your points stronger.

        Remove any logical fallacies.

        You should respond directly to your opponent's points.
        
        Include citations from your research.

        Use strong rhetoric, but concrete examples.
        """

    def _get_information_assimilation_prompt(
        self,
        beliefs: str,
        new_information: str,
        last_message_from_opponent: str,
    ):
        return f"""
        You are a research analyst named {self.name}.

        You are preparing for a debate where you must make the following argument:

        {beliefs}
           
        Here is what you have learned so far in your research:

        {new_information}

        This is the latest remarks from your opponent:

        `{last_message_from_opponent}`

        Please use this information to come up with an outline for your argument for the debate.

        Use only the information from your research, and explain your reasoning.
        """

    async def _build_questions_prompt(
        self, conversation: ConversationInterface
    ) -> Prompt:
        beliefs = self.knowledge_store.recall_recent(
            n_knowledge_items=self.n_knowledge_items_per_prompt,
            include_sources=["Strongly held beliefs"],
        )

        relevant_knowledge = self.knowledge_store.recall_recent(
            n_knowledge_items=self.n_knowledge_items_per_prompt,
            exclude_sources=["Strongly held beliefs"],
        )

        messages = [
            PromptMessage(
                role="system",
                content=self._get_question_prompt_text(
                    self.knowledge_store.knowledge_as_string(beliefs),
                ),
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
                content=f"""
                Here is what I've learned from my research already:

                {self.knowledge_store.knowledge_as_string(relevant_knowledge)}

                Thinking about this, a question I'd want to answer to strengthen my argument would be:
                """,
            ),
        ]

        return Prompt(messages=messages)

    async def _build_information_assimilation_prompt(
        self, conversation: ConversationInterface
    ) -> Prompt:
        beliefs = self.knowledge_store.recall_recent(
            n_knowledge_items=self.n_knowledge_items_per_prompt,
            include_sources=["Strongly held beliefs"],
        )

        relevant_knowledge = self.knowledge_store.recall_recent(
            n_knowledge_items=self.n_knowledge_items_per_prompt,
            exclude_sources=["Strongly held beliefs"],
        )

        messages = [
            PromptMessage(
                role="system",
                content=self._get_information_assimilation_prompt(
                    self.knowledge_store.knowledge_as_string(beliefs),
                    self.knowledge_store.knowledge_as_string(
                        relevant_knowledge
                    ),
                    conversation[-1].text,
                ),
            ),
            PromptMessage(
                role="assistant",
                content=f"""As {self.name}, an analyst trying to argue that:

                {beliefs}

                An outline of an argument that could win the debate would look like:
                """,
            ),
        ]

        return Prompt(messages=messages)

    async def _build_completion_prompt(
        self, conversation: ConversationInterface
    ) -> Prompt:
        argument_plan_prompt = (
            await self._build_information_assimilation_prompt(conversation)
        )
        argument_plan = (
            await self.client.get_completions(
                [argument_plan_prompt], **self._hyperparameters
            )
        )[0]
        logger.log_internal(f"{self.name} plans: {argument_plan}")

        beliefs = self.knowledge_store.recall_recent(
            n_knowledge_items=self.n_knowledge_items_per_prompt,
            include_sources=["Strongly held beliefs"],
        )

        relevant_knowledge = self.knowledge_store.recall_recent(
            n_knowledge_items=self.n_knowledge_items_per_prompt,
            exclude_sources=["Strongly held beliefs"],
        )

        messages = [
            PromptMessage(
                role="system",
                content=self._get_initial_prompt_text(
                    self.knowledge_store.knowledge_as_string(beliefs),
                ),
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
                content=f"""As {self.name}, an analyst trying to argue that:

                {beliefs}

                This is my outline for how I plan to respond to my opponent:

                {argument_plan}

                And here are my notes from my research:

                {relevant_knowledge}

                I will now execute my plan while filling in details, and respond with:
                """,
            ),
        ]

        first_attempt = (
            await self.client.get_completions(
                [Prompt(messages=messages)], **self._hyperparameters
            )
        )[0]

        logger.log_internal(
            f"{self.name} thinks about saying: {first_attempt}"
        )

        final_messages = [
            PromptMessage(
                role="system",
                content=self._get_reflection_prompt(
                    self.knowledge_store.knowledge_as_string(beliefs),
                ),
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
                content=f"""As {self.name}, an analyst trying to argue that:

                {beliefs}
                
                This is my outline for how I plan to respond to my opponent:

                {argument_plan}

                And here are my notes from my research:

                {relevant_knowledge}

                And here is my first draft of my response:

                {first_attempt}

                I will now revise my first draft to make my argument stronger and more concrete:
                """,
            ),
        ]

        return Prompt(messages=final_messages)
