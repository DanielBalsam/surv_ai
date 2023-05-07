import re
from core.conversation.interfaces import ConversationInterface

from lib.language.interfaces import (
    LargeLanguageModelClientInterface,
    Prompt,
    PromptMessage,
)
from lib.agent_log import agent_log
from .interfaces import (
    ToolbeltInterface,
    ToolInterface,
    NoMemoriesFoundException,
)


class Toolbelt(ToolbeltInterface):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        tools: list[ToolInterface],
    ):
        # if tools:
        #     tools.append(NoopTool(client))

        self.client = client
        self.tools = tools

    @staticmethod
    def tools_as_list(tools: list[ToolInterface]) -> str:
        return (
            "\n".join(f"{i + 1}. {t.instruction}" for i, t in enumerate(tools))
            if tools
            else ""
        )

    def _get_toolbelt_prompt(
        self, original_prompt: str, conversation: str
    ) -> str:
        return Prompt(
            messages=[
                PromptMessage(
                    role="system",
                    content=f"""A user will prompt you with a statement.

                    You should conduct research to assess this statement.
                    
                    These are the commands available to you:

                    {self.tools_as_list(self.tools)}

                    You MUST respond with a command.
                    """,
                ),
                PromptMessage(
                    role="user",
                    content=f"""{original_prompt}""",
                ),
                PromptMessage(
                    role="assistant",
                    content=f"""This is what the user has been talking about:
                    
                    ```
                    {conversation}
                    ```
                    
                    With that in mind, I will execute the following command:
                    """,
                ),
            ]
        )

    async def inspect(
        self,
        original_prompt: str,
        conversation: ConversationInterface,
        attempt=1,
    ):
        prompt = self._get_toolbelt_prompt(
            original_prompt, conversation.as_string()
        )

        response = (
            await self.client.get_completions([prompt], **{"temperature": 0.7})
        )[0]

        for tool in self.tools:
            match = re.match(tool.command, response)

            if match:
                memories = []
                for args in match.groups():
                    relevant_context = conversation.as_string()

                    agent_log.thought(
                        f"...Agent is using tool {tool.__class__.__name__} with args {args}..."
                    )

                    memories += await tool.use(
                        original_prompt, relevant_context, args
                    )

                    if not memories and attempt < 3:
                        memories = await self.inspect(
                            original_prompt, conversation, attempt=attempt + 1
                        )
                    elif not memories:
                        raise NoMemoriesFoundException()

                return memories

        return []
