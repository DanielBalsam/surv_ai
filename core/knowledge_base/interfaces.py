from typing import Protocol
from core.agent.interfaces import AgentInterface


class KnowledgeBaseInterface(Protocol):
    @property
    def knowledge(self) -> list[str]:
        ...

    def teach_to_agent(self, agent: AgentInterface):
        ...
