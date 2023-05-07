from thefuzz import process, fuzz
import asyncio
from core.memory_store.interfaces import Memory

from core.tools.interfaces import NoMemoriesFoundException, ToolbeltInterface
from .interfaces import AssemblyInterface, AssemblyResponse
from core.agent.research import ResearchAgent
from core.agent.binary import BinaryAgent
from core.conversation.conversation import Conversation


class VotingResearcherAssembly(AssemblyInterface):
    def __init__(
        self,
        client,
        toolbelt: ToolbeltInterface,
        n_agents=10,
        max_concurrency=10,
    ):
        self.client = client
        self.toolbelt = toolbelt

        self.n_agents = n_agents
        self.max_concurrency = max_concurrency

    async def _conduct_research(
        self,
        prompt: str,
        summaries: Conversation,
        relevant_articles: list[Memory],
    ):
        try:
            research_agent = ResearchAgent(
                self.client,
                toolbelt=self.toolbelt,
                n_memories_per_prompt=20,
                _hyperparameters={"temperature": 0.4},
            )

            for article in relevant_articles:
                await research_agent.memory_store.add_memory(article)

            response_conversation = Conversation()
            response = await research_agent.prompt(
                prompt, response_conversation
            )

            response_conversation.add(
                response, "Researcher", research_agent._color
            )

            binary_agent = BinaryAgent(
                self.client, _hyperparameters={"temperature": 0}
            )

            decision = await binary_agent.prompt(prompt, response_conversation)
            summaries.add(decision, "Decision", binary_agent._color)

            options = ["true", "false", "undecided"]
            coerced_decision = process.extractOne(
                decision, options, scorer=fuzz.partial_ratio
            )[0]

            return coerced_decision
        except NoMemoriesFoundException:
            return "undecided"
        except Exception:
            return "error"

    async def prompt(self, prompt: str):
        results = {"true": 0, "false": 0, "undecided": 0, "error": 0}

        summaries = Conversation()
        agents = 0

        relevant_articles = await self.toolbelt.inspect(prompt, Conversation())

        while agents < self.n_agents:
            coroutines = []

            error_rate = results["error"] / self.n_agents
            if error_rate > 0.25:
                raise Exception(
                    "Agent error rate is unusually high, likely an issue with API access."
                )

            for _ in range(min(self.max_concurrency, self.n_agents - agents)):
                coroutines.append(
                    self._conduct_research(
                        prompt, summaries, relevant_articles
                    )
                )
                agents += 1

            decisions = await asyncio.gather(*coroutines)

            for decision in decisions:
                results[decision] += 1

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
            error=results["error"],
            percent_in_favor=percent_in_favor,
            error_bar=error_bars,
            summaries=[],
        )
