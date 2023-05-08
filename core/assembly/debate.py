import asyncio
from random import sample

from pydantic import BaseModel
from core.conversation.conversation import Conversation
from core.tools.interfaces import ToolbeltInterface
from lib.language.interfaces import LargeLanguageModelClientInterface
from .interfaces import AssemblyInterface, AssemblyResponse

from thefuzz import fuzz, process

from core.agent.debate.analyst import AnalystAgent
from core.agent.debate.select_winner import SelectWinnerAgent
from core.agent.debate.team_lead import TeamLeadAgent
from core.agent.invertor import InvertorAgent
from core.knowledge_store.interfaces import Knowledge
from lib.agent_log import agent_log


class _TeamResult(BaseModel):
    points_in_favor: int
    points_against: int
    points_undecided: int
    summary: list[str]


class _DebateTeam:
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        rounds: int = 3,
        exchanges_per_round: int = 2,
    ):
        self.client = client
        self.max_conversation_length = rounds * exchanges_per_round

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
            name="Hank",
            _hyperparameters={
                "frequency_penalty": 0.5,
                "presence_penalty": 0.2,
                "temperature": 0.4,
            },
        )
        believer_agent.teach_text(assertion, source="Strongly held beliefs")

        debate.add(
            f"I believe {assertion}",
            believer_agent.name,
            believer_agent.color,
        )

        skeptic_agent = AnalystAgent(
            self.client,
            name="Sabrina",
            _hyperparameters={
                "frequency_penalty": 0.5,
                "presence_penalty": 0.2,
                "temperature": 0.4,
            },
        )

        skeptic_agent.teach_text(
            inverted_assertion, source="Strongly held beliefs"
        )

        debate.add(
            f"I believe {inverted_assertion}",
            skeptic_agent.name,
            skeptic_agent.color,
        )

        team_lead_agent = TeamLeadAgent(
            self.client,
            n_knowledge_items_per_prompt=5,
            name="Dan",
            _hyperparameters={
                "frequency_penalty": 0.5,
                "presence_penalty": 0.2,
                "temperature": 0.4,
            },
        )

        return believer_agent, skeptic_agent, team_lead_agent

    async def prompt(
        self, assertion: str, relevant_articles: list[Knowledge]
    ) -> _TeamResult:
        debate = Conversation()
        team_lead_thoughts = Conversation()

        agent_log.log_context(
            f'Initiating a debate about whether "{assertion}" is true.'
        )

        invertor = InvertorAgent(self.client)
        inverted_assertion = await invertor.prompt(assertion)

        (
            believer_agent,
            skeptic_agent,
            team_lead_agent,
        ) = await self._init_agents(assertion, inverted_assertion, debate)

        for article in relevant_articles:
            believer_agent.teach_knowledge(article)
            skeptic_agent.teach_knowledge(article)
            team_lead_agent.teach_knowledge(article)

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
                    color=debate_agent.color,
                )

                if dialogue_count % 2 == 0 and dialogue_count > 0:
                    summary = await team_lead_agent.prompt(
                        latest_assertion, conversation=debate
                    )
                    team_lead_thoughts.add(
                        summary,
                        team_lead_agent.name,
                        color=team_lead_agent.color,
                    )
                    summaries.append(summary)

                    decision = await self._get_decision_from_summary(
                        summary,
                        believer_agent,
                        skeptic_agent,
                    )

                    agent_log.log_context(f"Point awarded to {decision}")

                    if believer_agent.name in decision:
                        points_in_favor += 1
                    elif skeptic_agent.name in decision:
                        points_against += 1
                    else:
                        points_undecided += 1

        return _TeamResult(
            points_in_favor=points_in_favor,
            points_against=points_against,
            points_undecided=points_undecided,
            summary=summaries,
        )


class DebateAssembly(AssemblyInterface):
    def __init__(
        self,
        client,
        toolbelt: ToolbeltInterface,
        n_rounds=5,
        max_concurrency=10,
        max_articles_per_round=10,
    ):
        self.client = client
        self.toolbelt = toolbelt

        self.n_rounds = n_rounds
        self.max_concurrency = max_concurrency
        self.max_articles_per_round = max_articles_per_round

    async def prompt(self, prompt: str):
        results = []

        relevant_articles = await self.toolbelt.inspect(prompt, Conversation())

        round = 0

        while round < self.n_rounds:
            coroutines = []

            for _ in range(min(self.max_concurrency, self.n_rounds - round)):
                team = _DebateTeam(self.client)
                coroutines.append(
                    team.prompt(
                        prompt,
                        sample(relevant_articles, self.max_articles_per_round),
                    )
                )

            results += await asyncio.gather(*coroutines)

            round += self.max_concurrency

        points_in_favor = sum([r.points_in_favor for r in results])
        points_against = sum([r.points_against for r in results])
        points_undecided = sum([r.points_undecided for r in results])

        percent_in_favor = points_in_favor / (points_in_favor + points_against)
        uncertainty = points_undecided / (points_in_favor + points_against)

        return AssemblyResponse(
            percent_in_favor=percent_in_favor,
            uncertainty=uncertainty,
            error=0,
            summaries=[r.summary for r in results],
            in_favor=points_in_favor,
            against=points_against,
            undecided=points_undecided,
        )
