import asyncio
import logging
from random import sample

from core.agent.binary import BinaryAgent
from core.agent.research import ResearchAgent
from core.conversation.conversation import Conversation
from core.knowledge_store.interfaces import Knowledge
from core.tools.interfaces import NoMemoriesFoundException, ToolbeltInterface

from .interfaces import AssemblyInterface, AssemblyResponse


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
                relevant_articles, self.max_articles_per_agent
            ):
                research_agent.teach_knowledge(article)

            response_conversation = Conversation()
            response = await research_agent.prompt(
                prompt, response_conversation
            )

            response_conversation.add(
                response, "Researcher", research_agent.color
            )

            binary_agent = BinaryAgent(
                self.client, _hyperparameters={"temperature": 0}
            )

            decision = await binary_agent.prompt(prompt, response_conversation)
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
            logging.error(e)
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
