import asyncio
from random import sample
from typing import Optional

from surv_ai.lib.conversation.conversation import Conversation
from surv_ai.lib.knowledge_store.interfaces import Knowledge
from surv_ai.lib.llm.interfaces import LargeLanguageModelClientInterface
from surv_ai.lib.log import logger
from surv_ai.lib.tools.interfaces import NoMemoriesFoundException, ToolBeltInterface

from .agents.binary import BinaryAgent
from .agents.reasoning import ReasoningAgent
from .interfaces import SurveyInterface, SurveyResponse


class Survey(SurveyInterface):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        tool_belt: ToolBeltInterface,
        n_agents=10,
        max_concurrency=10,
        max_knowledge_per_agent=3,
        base_knowledge: Optional[list[Knowledge]] = None,
    ):
        self.client = client
        self.tool_belt = tool_belt

        self.n_agents = n_agents
        self.max_concurrency = max_concurrency
        self.max_knowledge_per_agent = max_knowledge_per_agent
        self.base_knowledge = base_knowledge

    async def _poll_agent(
        self,
        statement: str,
        summaries: Conversation,
        relevant_articles: list[Knowledge],
        index: int,
    ):
        try:
            reasoning_agent = ReasoningAgent(
                self.client,
                n_knowledge_items_per_prompt=self.max_knowledge_per_agent,
                _hyperparameters={"temperature": 0.6},
            )

            for article in sample(
                relevant_articles,
                min(len(relevant_articles), self.max_knowledge_per_agent),
            ):
                reasoning_agent.teach_knowledge(article)

            if self.base_knowledge:
                for knowledge in self.base_knowledge:
                    reasoning_agent.teach_knowledge(knowledge)

            response = await reasoning_agent.prompt(statement)

            binary_agent = BinaryAgent(
                self.client,
                _hyperparameters={"temperature": 0.2, "max_tokens": 5},
            )
            binary_agent.teach_text(statement, "Assertion")

            response_conversation = Conversation()
            response_conversation.add(response, f"Researcher #{index}", reasoning_agent.color)

            decision = await binary_agent.prompt(response_conversation)
            summaries.add(decision, f"Researcher #{index}", reasoning_agent.color)

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

    async def conduct(self, hypothesis: str):
        results = {"true": 0, "false": 0, "undecided": 0, "error": 0}

        summaries = Conversation()
        agents = 0

        try:
            relevant_articles = await self.tool_belt.inspect(self.client, hypothesis, self.base_knowledge or [])
        except NoMemoriesFoundException:
            return SurveyResponse(
                in_favor=0,
                against=0,
                undecided=0,
                error=self.n_agents,
                percent_in_favor=0,
                uncertainty=0,
            )

        while agents < self.n_agents:
            coroutines = []

            error_rate = results["error"] / self.n_agents
            if error_rate > 0.25:
                raise Exception("Agent error rate is unusually high, likely an issue with API access.")

            for index in range(min(self.max_concurrency, self.n_agents - agents)):
                coroutines.append(self._poll_agent(hypothesis, summaries, relevant_articles, index))
                agents += 1

            decisions = await asyncio.gather(*coroutines)

            for decision in decisions:
                results[decision] += 1

        if results["true"] + results["false"] == 0:
            percent_in_favor = 0
            uncertainty = 1
        else:
            percent_in_favor = results["true"] / (results["true"] + results["false"])
            uncertainty = results["undecided"] / (results["true"] + results["false"])

        return SurveyResponse(
            in_favor=results["true"],
            against=results["false"],
            undecided=results["undecided"],
            error=results["error"],
            percent_in_favor=percent_in_favor,
            uncertainty=uncertainty,
        )
