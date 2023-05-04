from core.conversation.conversation import Conversation
from .interfaces import TeamInterface, TeamResult
from .base import BaseTeam
from thefuzz import process, fuzz


from core.agent.debate.analyst import AnalystAgent
from core.agent.debate.select_winner import SelectWinnerAgent
from core.agent.invertor import InvertorAgent
from core.agent.debate.team_lead import TeamLeadAgent
from core.knowledge_base.common_sense import CommonSense
from lib.agent_log import agent_log


class DebateTeam(BaseTeam, TeamInterface):
    async def _get_decision_from_summary(
        self,
        summary: str,
        believer_agent: AnalystAgent,
        skeptic_agent: AnalystAgent,
    ) -> str:
        select_winner = SelectWinnerAgent(self.client)

        believer_agent_in_response = believer_agent.name in summary
        skeptic_agent_in_response = skeptic_agent.name in summary

        if believer_agent_in_response and not skeptic_agent_in_response:
            return believer_agent.name
        elif skeptic_agent_in_response and not believer_agent_in_response:
            return skeptic_agent.name

        decision = await select_winner.prompt(summary)

        options = [
            f"{believer_agent.name}",
            f"{skeptic_agent.name}",
            "Undecided",
        ]
        coerced_decision = process.extractOne(
            decision, options, scorer=fuzz.partial_ratio
        )[0]

        return coerced_decision

    async def _init_agents(
        self, assertion: str, inverted_assertion: str, debate: Conversation
    ):
        believer_agent = AnalystAgent(
            self.client,
            toolbelt=self.toolbelt,
            name="Hank",
            _hyperparameters={
                "frequency_penalty": 0.5,
                "presence_penalty": 0.2,
                "temperature": 0.4,
            },
        )
        await believer_agent.teach(assertion, source="Strongly held beliefs")
        await CommonSense().teach_to_agent(believer_agent)

        debate.add(
            f"I believe {assertion}",
            believer_agent.name,
            believer_agent._color,
        )

        skeptic_agent = AnalystAgent(
            self.client,
            toolbelt=self.toolbelt,
            name="Sabrina",
            _hyperparameters={
                "frequency_penalty": 0.5,
                "presence_penalty": 0.2,
                "temperature": 0.4,
            },
        )

        await skeptic_agent.teach(
            inverted_assertion, source="Strongly held beliefs"
        )
        await CommonSense().teach_to_agent(skeptic_agent)

        debate.add(
            f"I believe {inverted_assertion}",
            skeptic_agent.name,
            skeptic_agent._color,
        )

        team_lead_agent = TeamLeadAgent(
            self.client,
            toolbelt=self.toolbelt,
            n_memories_per_prompt=5,
            name="Dan",
            _hyperparameters={
                "frequency_penalty": 0.5,
                "presence_penalty": 0.2,
                "temperature": 0.4,
            },
        )

        return believer_agent, skeptic_agent, team_lead_agent

    async def _converse(self, assertion: str) -> TeamResult:
        debate = Conversation()
        team_lead_thoughts = Conversation()

        agent_log.info(
            f'{self.name} is starting a conversation about whether "{assertion}" is true.'
        )

        invertor = InvertorAgent(self.client)
        inverted_assertion = await invertor.prompt(assertion)

        (
            believer_agent,
            skeptic_agent,
            team_lead_agent,
        ) = await self._init_agents(assertion, inverted_assertion, debate)

        summaries = []
        points_in_favor = 0
        points_against = 0
        points_undecided = 0

        dialogue_count = 0
        while dialogue_count < self.max_conversation_length:
            for debate_agent, core_belief in zip(
                [believer_agent, skeptic_agent],
                [assertion, inverted_assertion],
            ):
                dialogue_count += 1

                latest_assertion = await debate_agent.prompt(
                    core_belief, conversation=debate
                )
                debate.add(
                    latest_assertion,
                    debate_agent.name,
                    color=debate_agent._color,
                )

                if dialogue_count % 2 == 0 and dialogue_count > 0:
                    summary = await team_lead_agent.prompt(
                        latest_assertion, conversation=debate
                    )
                    team_lead_thoughts.add(
                        summary,
                        team_lead_agent.name,
                        color=team_lead_agent._color,
                    )
                    summaries.append(summary)

                    decision = await self._get_decision_from_summary(
                        summary,
                        believer_agent,
                        skeptic_agent,
                    )

                    agent_log.info(f"Point awarded to {decision}")

                    if believer_agent.name in decision:
                        points_in_favor += 1
                    elif skeptic_agent.name in decision:
                        points_against += 1
                    else:
                        points_undecided += 1

        return TeamResult(
            points_in_favor=points_in_favor,
            points_against=points_against,
            points_undecided=points_undecided,
            summary=summaries,
        )
