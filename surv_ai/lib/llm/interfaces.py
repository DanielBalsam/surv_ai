from typing import Optional, Protocol

from pydantic import BaseModel


class PromptMessage(BaseModel):
    role: str
    content: str
    name: Optional[str]


class Prompt(BaseModel):
    messages: list[PromptMessage]


class LargeLanguageModelClientInterface(Protocol):
    def __init__(self, *args, **kwargs):
        ...

    async def get_completions(self, prompts: list[Prompt], **_hyperparameters) -> list[str]:
        ...
