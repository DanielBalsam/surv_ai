from typing import Optional, Protocol

from core.knowledge_store.interfaces import Knowledge, KnowledgeStoreInterface
from core.tools.interfaces import ToolbeltInterface
from lib.llm.interfaces import LargeLanguageModelClientInterface


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
