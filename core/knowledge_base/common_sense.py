from .interfaces import KnowledgeBaseInterface, AgentInterface
from datetime import datetime


class CommonSense(KnowledgeBaseInterface):
    @property
    def knowledge(self) -> list[str]:
        return [f"Today's date is {datetime.now().strftime('%B %d, %Y')}"]

    async def teach_to_agent(self, agent: AgentInterface):
        for knowledge in self.knowledge:
            await agent.teach(knowledge, source="Common sense")
