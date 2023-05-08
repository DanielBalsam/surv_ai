from typing import Optional, Protocol

from pydantic import BaseModel

from lib.llm.interfaces import LargeLanguageModelClientInterface


class Knowledge(BaseModel):
    text: str
    source: Optional[str]


class KnowledgeStoreInterface(Protocol):
    def __init__(
        self, client: LargeLanguageModelClientInterface, *args, **kwargs
    ):
        ...

    def recall_recent(self, n_knowledge_items=5, **kwargs) -> list[Knowledge]:
        ...

    def add_text(self, input: str, source: Optional[str] = None, **kwargs):
        ...

    def add_knowledge(self, knowledge: Knowledge, **kwargs):
        ...

    @staticmethod
    def knowledge_as_string(knowledge: list[Knowledge]) -> str:
        ...
