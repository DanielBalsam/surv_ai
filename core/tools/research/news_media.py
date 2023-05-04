from typing import Optional
from bs4 import BeautifulSoup
from enum import StrEnum
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


class MediaSources(StrEnum):
    TOP_10 = "top10"
    TOP_100 = "top100"


class NewsMediaTool(QueryToolInterface):
    instruction = """
        `NEWS(query)` - use keywords to search the New York Times for additional information.
    """
    command = r"NEWS\((.+)\)"

    _base_url = "https://api.goperigon.com/v1/all"

    UNPARSEABLE_SOURCES = ["wsj.com"]

    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        embeddings: Optional[EmbeddingInterface] = None,
        begin_date="2023-04-01",
        end_date="2023-05-01",
        source_group=MediaSources.TOP_10,
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
        self.source_group = source_group

    async def _search(self, session, query: str) -> list[str]:
        params = {
            "q": " AND ".join(re.sub(r"[^A-Za-z0-9 ]+", "", query).split(" ")),
            "from": f"{self.begin_date}",
            "to": f"{self.end_date}",
            "apiKey": os.getenv("PERIGON_API_KEY", ""),
            "sourceGroup": self.source_group,
            "language": "en",
            "paywwall": "false",
            "medium": "Article",
        }

        async with session.get(self._base_url, params=params) as response:
            data = await response.json()
            return [
                result
                for result in data["articles"]
                if result["source"]["domain"] not in self.UNPARSEABLE_SOURCES
            ]

    async def _get_page_text(self, _, web_url: str) -> str:
        if web_url in self._already_searched:
            return self._already_searched[web_url]

        data = subprocess.run(
            "curl -s " + web_url, shell=True, capture_output=True
        ).stdout.decode("utf-8")
        print(data)

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
        relevant_context: str,
        web_url: str,
    ):
        paragraphs = await self._get_page_text(session, web_url)

        most_relevant_paragraphs = (
            self.get_most_relevant_information_from_page(
                original_prompt,
                relevant_context,
                paragraphs,
            )
        )

        prompt = Prompt(
            messages=[
                PromptMessage(
                    content=f"""Please summarize the following text as it relates to the following query: '{original_prompt}'.
                    
                    Relevant context:

                    {relevant_context}
                    """,
                    role="system",
                ),
                PromptMessage(
                    content="\n".join(most_relevant_paragraphs),
                    role="user",
                ),
            ],
        )
        response = await self.client.get_completions([prompt])

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
        relevant_context: str,
        result: dict,
    ):
        agent_log.thought(
            f"......Retrieving {result['source']['domain']} page with headline {result['title']}......"
        )
        page_summary = await self._ingest_page_information(
            session,
            original_prompt,
            relevant_context,
            result["url"],
        )

        page_embeddings = self.embeddings.embed([page_summary])[0]
        return Memory(
            text=f"{result['source']['domain']} article entitled \"{result['title']}\": {page_summary}",
            source=result["url"],
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

            if len(search_results) == 0:
                return []

            return await asyncio.gather(
                *[
                    self._ingest_pages(
                        session, original_prompt, relevant_context, web_url
                    )
                    for web_url in search_results[: self.n_pages]
                ]
            )
