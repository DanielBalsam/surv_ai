from core.tools.interfaces import ToolbeltInterface
from ..interfaces import AssemblyInterface, AssemblyResponse
from core.teams.debate import DebateTeam


class WinnerTakeAllDebateAssembly(AssemblyInterface):
    def __init__(self, client, toolbelt: ToolbeltInterface, n_teams=5):
        self.client = client
        self.toolbelt = toolbelt

        self.n_teams = n_teams

    async def prompt(self, input: str):
        results = []

        for team in range(self.n_teams):
            team = DebateTeam(self.client, toolbelt=self.toolbelt)
            results.append(await team.prompt(input))

        points_in_favor = sum(
            [1 if r.points_in_favor > r.points_against else 0 for r in results]
        )
        points_against = sum(
            [1 if r.points_in_favor < r.points_against else 0 for r in results]
        )
        points_undecided = sum(
            [
                1 if r.points_in_favor == r.points_against else 0
                for r in results
            ]
        )

        percent_in_favor = points_in_favor / (points_in_favor + points_against)
        error_bars = points_undecided / (points_in_favor + points_against)

        return AssemblyResponse(
            percent_in_favor=percent_in_favor,
            error_bar=error_bars,
            summaries=[r.summary for r in results],
            in_favor=points_in_favor,
            against=points_against,
            undecided=points_undecided,
        )
