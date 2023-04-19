from .interfaces import TeamInterface
from .base import BaseTeam

from core.agent.research import ResearchAgent
from core.agent.decision import DecisionAgent
from core.agent.moderator import ModeratorAgent
from core.agent.skeptic import SkepticAgent


class EstimatorTeam(BaseTeam, TeamInterface):
    async def _converse(self, input: str) -> str:
        conversation = 0

        research_agent = ResearchAgent(self.client, toolbelt=self.toolbelt)
        skeptic_agent = SkepticAgent(self.client, toolbelt=self.toolbelt)

        while conversation < self.n_conversations:
            moderator_agent = ModeratorAgent(self.client)
            dialogue_count = 0

            query = input
            while dialogue_count < self.max_conversation_length:
                assertion = await research_agent.prompt(query)
                print(f"Research agent:", assertion)

                query = await skeptic_agent.prompt(assertion)
                print(f"Skeptic agent:", query)

                await moderator_agent.teach(assertion, source="research agent")
                await moderator_agent.teach(query, source="skeptic agent")

                if dialogue_count % 6 == 4:
                    moderator_result = await moderator_agent.prompt(input)
                    print(f"Moderator agent:", moderator_result)

                    if "off topic" in moderator_result.lower():
                        break

                dialogue_count += 2

            conversation += 1

        decision_agent = DecisionAgent(self.client)
        await decision_agent.transfer_memories(research_agent)

        return await decision_agent.prompt(input)
