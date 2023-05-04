from typing import Optional
from bs4 import BeautifulSoup
import re
import os
import asyncio
import subprocess

from aiohttp import ClientSession
from lib.vector.search import VectorSearch, VectorSearchType
from core.embeddings.interfaces import EmbeddingInterface
from core.embeddings.sbert import SentenceBertEmbedding
from core.memory_store.interfaces import Memory

from lib.language.interfaces import (
    LargeLanguageModelClientInterface,
    Prompt,
    PromptMessage,
)
from lib.agent_log import agent_log

from .interfaces import QueryToolInterface


class NewYorkTimesTool(QueryToolInterface):
    instruction = """
        `NEW_YORK_TIMES(*keywords)` - search the New York Times using no more than three keywords.
    """
    command = r"NEW_YORK_TIMES\((.+)\)"

    _base_url = "https://api.nytimes.com/svc/search/v2/"

    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        embeddings: Optional[EmbeddingInterface] = None,
        begin_date="20230401",
        end_date="20230501",
        n_pages=1,
    ):
        if not embeddings:
            embeddings = SentenceBertEmbedding()

        self.client = client
        self.n_pages = n_pages
        self.embeddings = embeddings

        self._already_searched = dict()

        self.begin_date = begin_date
        self.end_date = end_date

    async def _search(self, session, query: str, attempt=1) -> list[str]:
        params = {
            "q": re.sub(r"[^A-Za-z0-9 ]+", "", query),
            "begin_date": self.begin_date,
            "end_date": self.end_date,
            "api-key": os.getenv("NEW_YORK_TIMES_KEY_ID", ""),
        }
        async with session.get(
            self._base_url + "/articlesearch.json", params=params
        ) as response:
            try:
                data = await response.json()
                return [result for result in data["response"]["docs"]]
            except Exception:
                if attempt < 3:
                    return await self._search(session, query)

    async def _get_page_text(self, _, web_url: str) -> str:
        if web_url in self._already_searched:
            return self._already_searched[web_url]

        data = subprocess.run(
            "curl -s " + web_url, shell=True, capture_output=True
        ).stdout.decode("utf-8")

        soup = BeautifulSoup(data, "html.parser")

        results = []
        for paragraph in soup.find_all("p"):
            results.append(paragraph.text)

        self._already_searched[web_url] = results

        return results

    async def _ingest_page_information(
        self,
        session,
        original_prompt: str,
        headline: str,
        web_url: str,
    ):
        paragraphs = await self._get_page_text(session, web_url)

        prompt = Prompt(
            messages=[
                PromptMessage(
                    content=f"""A user has asked you some questions:
                    
                    {original_prompt}

                    All subsequent messages will be paragraphs from a New York Times article is entitled "{headline}."

                    Your job is to extract any useful information that might help someone answer these questions.

                    Remember to include as much data and concrete examples as possible from the article.

                    For each useful piece of information you extract, please state why it relates to the original questions.
                    """,
                    role="system",
                ),
                *[
                    PromptMessage(
                        content=paragraph, role="user", name="Article"
                    )
                    for paragraph in paragraphs
                ],
                PromptMessage(
                    role="assistant",
                    content=f"I have read the New York Times article entitled {headline} and some useful information is:",
                ),
            ],
        )
        response = await self.client.get_completions(
            [prompt], **{"temperature": 0.2}
        )

        return response[0]

    def get_most_relevant_information_from_page(
        self,
        query: str,
        relevant_context: str,
        paragraphs: list[str],
        n_results=10,
    ):
        prompt_embedding = self.embeddings.embed([query + relevant_context])[0]
        paragraph_embeddings = self.embeddings.embed(paragraphs)
        _, indices = VectorSearch.sort_by_similarity(
            paragraph_embeddings,
            prompt_embedding,
            type=VectorSearchType.L2,
        )

        return [paragraphs[i] for i in indices[0:n_results]]

    async def _ingest_pages(
        self,
        session: ClientSession,
        original_prompt: str,
        result: dict,
    ):
        agent_log.thought(
            f"......Retrieving NYT page with headline {result['headline']['main']}......"
        )
        page_summary = await self._ingest_page_information(
            session,
            original_prompt,
            result["headline"]["main"],
            result["web_url"],
        )

        page_embeddings = self.embeddings.embed([page_summary])[0]
        return Memory(
            text=f"New York Times article entitled \"{result['headline']['main']}\": {page_summary}",
            source=result["web_url"],
            embedding=page_embeddings,
        )

    async def use(
        self,
        original_prompt: str,
        relevant_context: str,
        search_query: str,
    ) -> list[Memory]:
        async with ClientSession() as session:
            search_results = await self._search(session, search_query)

            if not search_results:
                return []

            return await asyncio.gather(
                *[
                    self._ingest_pages(session, original_prompt, result)
                    for result in search_results[: self.n_pages]
                ]
            )
