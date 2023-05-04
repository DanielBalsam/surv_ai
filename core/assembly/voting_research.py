from thefuzz import process, fuzz

from core.tools.interfaces import ToolbeltInterface
from .interfaces import AssemblyInterface, AssemblyResponse
from core.agent.research import ResearchAgent
from core.agent.binary import BinaryAgent
from core.conversation.conversation import Conversation


class VotingResearcherAssembly(AssemblyInterface):
    def __init__(self, client, toolbelt: ToolbeltInterface, n_agents=50):
        self.client = client
        self.toolbelt = toolbelt

        self.n_agents = n_agents

    async def prompt(self, input: str):
        options = ["true", "false", "undecided"]
        results = {"true": 0, "false": 0, "undecided": 0}

        summaries = Conversation()
        for _ in range(self.n_agents):
            research_agent = ResearchAgent(
                self.client,
                toolbelt=self.toolbelt,
                n_memories_per_prompt=10,
                _hyperparameters={"temperature": 0.2},
            )

            response = await research_agent.prompt(input, Conversation())

            summaries.add(response, "Researcher", research_agent._color)

            binary_agent = BinaryAgent(
                self.client, _hyperparameters={"temperature": 0}
            )
            await binary_agent.teach(response)

            decision = await binary_agent.prompt(input)
            summaries.add(decision, "Decision", binary_agent._color)

            coerced_decision = process.extractOne(
                decision, options, scorer=fuzz.partial_ratio
            )[0]

            results[coerced_decision] += 1

        percent_in_favor = results["true"] / (
            results["true"] + results["false"]
        )
        error_bars = results["undecided"] / (
            results["true"] + results["false"]
        )

        return AssemblyResponse(
            in_favor=results["true"],
            against=results["false"],
            undecided=results["undecided"],
            percent_in_favor=percent_in_favor,
            error_bar=error_bars,
            summaries=[[summary.text] for summary in summaries],
        )
