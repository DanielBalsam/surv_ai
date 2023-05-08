import asyncio
from random import sample
from core.conversation.conversation import Conversation
from core.tools.interfaces import ToolbeltInterface
from .interfaces import AssemblyInterface, AssemblyResponse
from core.teams.debate import DebateTeam


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
                team = DebateTeam(self.client)
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
            summaries=[r.summary for r in results],
            in_favor=points_in_favor,
            against=points_against,
            undecided=points_undecided,
        )
