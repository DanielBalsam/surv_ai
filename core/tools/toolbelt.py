import re

from lib.language.interfaces import (
    LargeLanguageModelClientInterface,
    Prompt,
    PromptMessage,
)
from core.memory_store.interfaces import MemoryStoreInterface
from .interfaces import ToolbeltInterface, ToolInterface
from .noop import NoopTool


class Toolbelt(ToolbeltInterface):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        tools: list[ToolInterface],
    ):
        tools.append(NoopTool(client))

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
        self, original_prompt: str, memory_list: str
    ) -> str:
        return Prompt(
            messages=[
                PromptMessage(
                    role="system",
                    content=f"""A user will ask you a question.
                    
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
                    content=f"""This is what I already know:
                    
                    {memory_list}
                    
                    With that in mind, I will execute the following command:
                    """,
                ),
            ]
        )

    async def inspect(
        self,
        original_prompt: str,
        memory_store: MemoryStoreInterface,
    ):
        relevant_memories = await memory_store.recall(original_prompt)

        prompt = self._get_toolbelt_prompt(
            original_prompt, memory_store.memories_as_list(relevant_memories)
        )

        response = (await self.client.get_completions([prompt]))[0]

        for tool in self.tools:
            match = re.match(tool.command, response)

            if match:
                for args in match.groups():
                    relevant_context = memory_store.memories_as_list(
                        relevant_memories
                    )
                    memories = await tool.use(
                        original_prompt, relevant_context, args
                    )

                    if memories:
                        for memory in memories:
                            await memory_store.add_memory(memory)

                return True

        return False
