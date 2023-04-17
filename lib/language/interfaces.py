from typing import Protocol


from pydantic import BaseModel


class PromptMessage(BaseModel):
    role: str
    content: str


class Prompt(BaseModel):
    messages: list[PromptMessage]


class LargeLanguageModelClientInterface(Protocol):
    def __init__(self, *args, **kwargs):
        ...

    async def get_completions(self, prompts: list[Prompt]) -> list[str]:
        ...
