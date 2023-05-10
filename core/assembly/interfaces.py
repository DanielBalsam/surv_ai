from pydantic import BaseModel

from lib.llm.interfaces import LargeLanguageModelClientInterface
from typing import Optional, Protocol

from core.knowledge_store.interfaces import Knowledge, KnowledgeStoreInterface
from core.tools.interfaces import ToolbeltInterface


class AgentInterface(Protocol):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        knowledge_store: Optional[KnowledgeStoreInterface],
        toolbelt: Optional[list[ToolbeltInterface]],
    ):
        ...

    async def prompt(self, input: str) -> str:
        ...

    def teach_text(self, input: str, source: Optional[str] = "User"):
        ...

    def teach_knowledge(self, knowledge: Knowledge):
        ...


class AssemblyResponse(BaseModel):
    in_favor: int
    against: int
    undecided: int
    error: int

    percent_in_favor: float
    uncertainty: float
    summaries: list[list[str]]


class AssemblyInterface(Protocol):
    def __init__(
        self, client: LargeLanguageModelClientInterface, *args, **kwargs
    ):
        ...

    async def run(self, prompt: str) -> AssemblyResponse:
        ...
