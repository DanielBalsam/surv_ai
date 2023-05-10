import asyncio
from random import sample

from .agents.binary import BinaryAgent
from .agents.research import ResearchAgent
from core.conversation.conversation import Conversation
from core.knowledge_store.interfaces import Knowledge
from core.tools.interfaces import NoMemoriesFoundException, ToolbeltInterface

from lib.log import logger

from ..interfaces import AssemblyInterface, AssemblyResponse


class ResearchAssembly(AssemblyInterface):
    def __init__(
        self,
        client,
        toolbelt: ToolbeltInterface,
        n_agents=10,
        max_concurrency=10,
        max_articles_per_agent=5,
    ):
        self.client = client
        self.toolbelt = toolbelt

        self.n_agents = n_agents
        self.max_concurrency = max_concurrency
        self.max_articles_per_agent = max_articles_per_agent

    async def _conduct_research(
        self,
        prompt: str,
        summaries: Conversation,
        relevant_articles: list[Knowledge],
    ):
        try:
            research_agent = ResearchAgent(
                self.client,
                n_knowledge_items_per_prompt=self.max_articles_per_agent,
                _hyperparameters={"temperature": 0.4},
            )

            for article in sample(
                relevant_articles,
                min(len(relevant_articles), self.max_articles_per_agent),
            ):
                research_agent.teach_knowledge(article)

            response = await research_agent.complete(prompt)

            binary_agent = BinaryAgent(
                self.client,
                _hyperparameters={"temperature": 0.2, "max_tokens": 5},
            )
            binary_agent.teach_text(prompt, "Assertion")

            response_conversation = Conversation()
            response_conversation.add(
                response, "Researcher", research_agent.color
            )

            decision = await binary_agent.complete(response_conversation)
            summaries.add(decision, "Decision", binary_agent.color)

            true_in_decision = "true" in decision.lower()
            false_in_decision = "false" in decision.lower()

            if true_in_decision and not false_in_decision:
                return "true"
            elif false_in_decision and not true_in_decision:
                return "false"
            else:
                return "undecided"
        except NoMemoriesFoundException:
            return "undecided"
        except Exception as e:
            logger.log_exception(e)
            return "error"

    async def run(self, prompt: str):
        results = {"true": 0, "false": 0, "undecided": 0, "error": 0}

        summaries = Conversation()
        agents = 0

        relevant_articles = await self.toolbelt.inspect(prompt)

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

        if results["true"] + results["false"] == 0:
            percent_in_favor = 0
            uncertainty = 1
        else:
            percent_in_favor = results["true"] / (
                results["true"] + results["false"]
            )
            uncertainty = results["undecided"] / (
                results["true"] + results["false"]
            )

        return AssemblyResponse(
            in_favor=results["true"],
            against=results["false"],
            undecided=results["undecided"],
            error=results["error"],
            percent_in_favor=percent_in_favor,
            uncertainty=uncertainty,
            summaries=[[summary.text] for summary in summaries],
        )
