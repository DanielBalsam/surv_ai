import re
from typing import Optional

from surv_ai.lib.knowledge_store.interfaces import Knowledge
from surv_ai.lib.llm.interfaces import (
    LargeLanguageModelClientInterface,
    Prompt,
    PromptMessage,
)
from surv_ai.lib.log import logger

from .interfaces import (
    NoMemoriesFoundException,
    ToolBeltInterface,
    ToolInterface,
    ToolResult,
)


class ToolBelt(ToolBeltInterface):
    def __init__(
        self,
        tools: Optional[list[ToolInterface]] = None,
    ):
        self.tools = tools or []

    @staticmethod
    def tools_as_list(tools: list[ToolInterface]) -> str:
        return "\n".join(f"{i + 1}. {t.instruction}" for i, t in enumerate(tools)) if tools else ""

    def _get_tool_belt_prompt(self, original_prompt: str, base_knowledge: list[Knowledge]) -> str:
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
                *[
                    PromptMessage(
                        role="system",
                        content=f"This information may be relevant to your query: {knowledge.text}",
                    )
                    for knowledge in base_knowledge
                ],
                PromptMessage(
                    role="user",
                    content=f"""{original_prompt}""",
                ),
                PromptMessage(
                    role="assistant",
                    content="""
                    In order to research the user's prompt, I will execute the following command:
                    """,
                ),
            ]
        )

    async def inspect(
        self,
        client: LargeLanguageModelClientInterface,
        original_prompt: str,
        base_knowledge: list[Knowledge],
        attempt=1,
    ) -> list[ToolResult]:
        prompt = self._get_tool_belt_prompt(original_prompt, base_knowledge)

        response = (await client.get_completions([prompt], **{"temperature": 0.7}))[0].strip()

        results: list[ToolResult] = []
        for tool in self.tools:
            match = re.match(tool.command, response)

            if match:
                for args in match.groups():
                    logger.log_context(f"...Using tool: {response}...")

                    results += await tool.use(args)

                    if not results and attempt < 3:
                        results = await self.inspect(client, original_prompt, base_knowledge, attempt=attempt + 1)
                    elif not results:
                        raise NoMemoriesFoundException()

        return results
