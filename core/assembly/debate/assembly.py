import asyncio
from random import sample

from pydantic import BaseModel
from core.conversation.conversation import Conversation
from core.tools.interfaces import ToolbeltInterface
from lib.llm.interfaces import LargeLanguageModelClientInterface
from ..interfaces import AssemblyInterface, AssemblyResponse

from .agents.debater_agent import DebaterAgent
from .agents.select_winner import SelectWinnerAgent
from .agents.debate_moderator import DebateModeratorAgent
from .agents.statement_invertor import StatementInvertorAgent
from core.knowledge_store.interfaces import Knowledge
from lib.log import logger


class _TeamResult(BaseModel):
    points_in_favor: int
    points_against: int
    points_undecided: int
    error: int
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
        in_favor_debater_agent: DebaterAgent,
        against_debater_agent: DebaterAgent,
    ) -> str:
        select_winner = SelectWinnerAgent(
            self.client, _hyperparameters={"temperature": 0.2, "max_tokens": 5}
        )

        in_favor_debater_agent_in_response = (
            in_favor_debater_agent.name in summary
        )
        against_debater_agent_in_response = (
            against_debater_agent.name in summary
        )

        if (
            in_favor_debater_agent_in_response
            and not against_debater_agent_in_response
        ):
            return in_favor_debater_agent.name
        elif (
            against_debater_agent_in_response
            and not in_favor_debater_agent_in_response
        ):
            return against_debater_agent.name

        decision = await select_winner.complete(summary)

        in_favor_debater_agent_in_decision = (
            in_favor_debater_agent.name in decision
        )
        against_debater_agent_in_decision = (
            against_debater_agent.name in decision
        )

        if (
            in_favor_debater_agent_in_decision
            and not against_debater_agent_in_decision
        ):
            return in_favor_debater_agent.name
        elif (
            against_debater_agent_in_decision
            and not in_favor_debater_agent_in_decision
        ):
            return against_debater_agent.name

        return "Undecided"

    async def _init_agents(
        self, assertion: str, inverted_assertion: str, debate: Conversation
    ):
        in_favor_debater_agent = DebaterAgent(
            self.client,
            name="Hank",
            _hyperparameters={
                "frequency_penalty": 0.5,
                "presence_penalty": 0.2,
                "temperature": 0.4,
            },
        )
        in_favor_debater_agent.teach_text(
            assertion, source="Strongly held beliefs"
        )

        debate.add(
            f"I believe {assertion}",
            in_favor_debater_agent.name,
            in_favor_debater_agent.color,
        )

        against_debater_agent = DebaterAgent(
            self.client,
            name="Sabrina",
            _hyperparameters={
                "frequency_penalty": 0.5,
                "presence_penalty": 0.2,
                "temperature": 0.4,
            },
        )

        against_debater_agent.teach_text(
            inverted_assertion, source="Strongly held beliefs"
        )

        debate.add(
            f"I believe {inverted_assertion}",
            against_debater_agent.name,
            against_debater_agent.color,
        )

        moderator_agent = DebateModeratorAgent(
            self.client,
            n_knowledge_items_per_prompt=5,
            name="Dan",
            _hyperparameters={
                "frequency_penalty": 0.5,
                "presence_penalty": 0.2,
                "temperature": 0.4,
            },
        )
        moderator_agent.teach_text(
            assertion,
            source="Debate topic",
        )

        return in_favor_debater_agent, against_debater_agent, moderator_agent

    async def run(
        self, assertion: str, relevant_articles: list[Knowledge]
    ) -> _TeamResult:
        debate = Conversation()
        team_lead_thoughts = Conversation()

        logger.log_context(
            f'Initiating a debate about whether "{assertion}" is true.'
        )

        invertor = StatementInvertorAgent(self.client)
        inverted_assertion = await invertor.complete(assertion)

        (
            in_favor_debater_agent,
            against_debater_agent,
            moderator_agent,
        ) = await self._init_agents(assertion, inverted_assertion, debate)

        for article in relevant_articles:
            in_favor_debater_agent.teach_knowledge(article)
            against_debater_agent.teach_knowledge(article)
            moderator_agent.teach_knowledge(article)

        summaries = []
        points_in_favor = 0
        points_against = 0
        points_undecided = 0
        error = 0

        dialogue_count = 0
        while dialogue_count < self.max_conversation_length:
            for debate_agent in [
                in_favor_debater_agent,
                against_debater_agent,
            ]:
                dialogue_count += 1

                try:
                    latest_assertion = await debate_agent.complete(debate)
                    debate.add(
                        latest_assertion,
                        debate_agent.name,
                        color=debate_agent.color,
                    )

                    if dialogue_count % 2 == 0 and dialogue_count > 0:
                        summary = await moderator_agent.complete(debate)
                        team_lead_thoughts.add(
                            summary,
                            moderator_agent.name,
                            color=moderator_agent.color,
                        )
                        summaries.append(summary)

                        decision = await self._get_decision_from_summary(
                            summary,
                            in_favor_debater_agent,
                            against_debater_agent,
                        )

                        logger.log_context(f"Point awarded to {decision}")

                        if in_favor_debater_agent.name in decision:
                            points_in_favor += 1
                        elif against_debater_agent.name in decision:
                            points_against += 1
                        else:
                            points_undecided += 1
                except Exception:
                    logger.log_exception("Error encountered during debate")
                    error += 1

        return _TeamResult(
            points_in_favor=points_in_favor,
            points_against=points_against,
            points_undecided=points_undecided,
            error=error,
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

    async def run(self, prompt: str):
        results = []

        relevant_articles = await self.toolbelt.inspect(prompt)

        round = 0

        while round < self.n_rounds:
            coroutines = []

            for _ in range(min(self.max_concurrency, self.n_rounds - round)):
                team = _DebateTeam(self.client)
                coroutines.append(
                    team.run(
                        prompt,
                        sample(
                            relevant_articles,
                            min(
                                len(relevant_articles),
                                self.max_articles_per_round,
                            ),
                        ),
                    )
                )

            results += await asyncio.gather(*coroutines)

            round += self.max_concurrency

        points_in_favor = sum([r.points_in_favor for r in results])
        points_against = sum([r.points_against for r in results])
        points_undecided = sum([r.points_undecided for r in results])
        error = sum([r.error for r in results])

        percent_in_favor = points_in_favor / (points_in_favor + points_against)
        uncertainty = points_undecided / (points_in_favor + points_against)

        return AssemblyResponse(
            percent_in_favor=percent_in_favor,
            uncertainty=uncertainty,
            error=error,
            summaries=[r.summary for r in results],
            in_favor=points_in_favor,
            against=points_against,
            undecided=points_undecided,
        )
