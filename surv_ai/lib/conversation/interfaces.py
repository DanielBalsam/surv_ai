from typing import Protocol

from pydantic import BaseModel

from ..llm.interfaces import PromptMessage


class ChatMessage(BaseModel):
    text: str
    speaker: str
    color: str

    def __str__(self):
        return f'{self.speaker} said, "{self.text}"'


class ConversationInterface(Protocol):
    def __init__(self, *args, **kwargs):
        ...

    def add(self, message: PromptMessage):
        ...
