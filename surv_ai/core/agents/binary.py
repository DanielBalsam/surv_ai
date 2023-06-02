from surv_ai.lib.conversation.interfaces import ConversationInterface
from surv_ai.lib.llm.interfaces import Prompt, PromptMessage

from ..agent import BaseAgent


class BinaryAgent(BaseAgent):
    def _get_initial_prompt_text(self, assertion: str):
        return f"""        
        Your job is to determine if the user believes a hypothesis is more likely to be true or false.

        Here is the original hypothesis:
        
        "{assertion}"

        The next message will be what the user thinks.

        Your only options are "True," or "False".

        Please always respond with a single word.
        """

    async def _build_completion_prompt(self, conversation: ConversationInterface) -> str:
        assertion = self.knowledge_store.recall_recent(
            n_knowledge_items=1,
            include_sources=["Assertion"],
        )[0]

        self.messages = [
            PromptMessage(
                role="system",
                content=self._get_initial_prompt_text(assertion),
            ),
            PromptMessage(
                role="user",
                content=conversation.as_string(),
                name="User thinks"
            ),
            PromptMessage(
                role="assistant",
                content="I believe the user thinks this hypothesis is: ",
            ),
        ]

        return Prompt(messages=self.messages)

    async def prompt(self, statement: str, *args, **kwargs) -> str:
        prompt = await self._build_completion_prompt(statement, *args, **kwargs)

        response = (await self.client.get_completions([prompt], **self._hyperparameters))[0]

        return response
