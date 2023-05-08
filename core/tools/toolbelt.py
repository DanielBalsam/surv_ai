import re

from lib.log import logger
from lib.llm.interfaces import (
    LargeLanguageModelClientInterface,
    Prompt,
    PromptMessage,
)

from .interfaces import (
    NoMemoriesFoundException,
    ToolbeltInterface,
    ToolInterface,
)


class Toolbelt(ToolbeltInterface):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        tools: list[ToolInterface],
    ):
        self.client = client
        self.tools = tools

    @staticmethod
    def tools_as_list(tools: list[ToolInterface]) -> str:
        return (
            "\n".join(f"{i + 1}. {t.instruction}" for i, t in enumerate(tools))
            if tools
            else ""
        )

    def _get_toolbelt_prompt(self, original_prompt: str) -> str:
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
                    content=f"""
                    In order to research the user's prompt, I will execute the following command:
                    """,
                ),
            ]
        )

    async def inspect(
        self,
        original_prompt: str,
        attempt=1,
    ):
        prompt = self._get_toolbelt_prompt(original_prompt)

        response = (
            await self.client.get_completions([prompt], **{"temperature": 0.7})
        )[0]

        for tool in self.tools:
            match = re.match(tool.command, response)

            if match:
                knowledge = []
                for args in match.groups():
                    logger.log_context(
                        f"...Using tool {tool.__class__.__name__} with args {args}..."
                    )

                    knowledge += await tool.use(original_prompt, args)

                    if not knowledge and attempt < 3:
                        knowledge = await self.inspect(
                            original_prompt, attempt=attempt + 1
                        )
                    elif not knowledge:
                        raise NoMemoriesFoundException()

                return knowledge

        return []
