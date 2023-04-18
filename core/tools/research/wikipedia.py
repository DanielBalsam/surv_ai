from typing import Optional
from bs4 import BeautifulSoup
import re
import numpy as np
import asyncio

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

from .interfaces import QueryToolInterface


class WikipediaTool(QueryToolInterface):
    instruction = """
        `WIKIPEDIA(query)` - search Wikipedia for additional information.
    """
    command = r"WIKIPEDIA\((.+)\)"

    _base_url = "https://en.wikipedia.org/w/api.php"

    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        embeddings: Optional[EmbeddingInterface] = None,
        n_pages=1,
    ):
        if not embeddings:
            embeddings = SentenceBertEmbedding()
        self.client = client
        self.n_pages = n_pages
        self.embeddings = embeddings

    async def _search(self, session, query: str) -> list[str]:
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": re.sub(r"[^A-Za-z0-9 ]+", "", query),
        }
        async with session.get(self._base_url, params=params) as response:
            data = await response.json()
            return [result["title"] for result in data["query"]["search"]]

    async def _get_page_text(self, session, page_title: str) -> str:
        params = {
            "action": "parse",
            "format": "json",
            "page": page_title,
            "prop": "text",
        }
        async with session.get(self._base_url, params=params) as response:
            data = await response.json()
            html_content = data["parse"]["text"]["*"]

            soup = BeautifulSoup(html_content, "html.parser")

            results = []
            for paragraph in soup.find_all("p"):
                results.append(paragraph.text)

            return results

    async def _ingest_page_information(
        self,
        session,
        original_prompt: str,
        relevant_context: str,
        page_title: str,
    ):
        paragraphs = await self._get_page_text(session, page_title)

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
            type=VectorSearchType.COSINE,
        )

        return [paragraphs[i] for i in indices[0:n_results]]

    async def _ingest_pages(
        self,
        session: ClientSession,
        original_prompt: str,
        relevant_context: str,
        page_title: str,
    ):
        page_summary = await self._ingest_page_information(
            session,
            original_prompt,
            relevant_context,
            page_title,
        )

        source = (
            f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
        )

        page_embeddings = self.embeddings.embed([page_summary])[0]
        return Memory(
            text=f"You searched the Wikipedia page entitled {page_title}: {page_summary}",
            source=source,
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
                        session, original_prompt, relevant_context, page_title
                    )
                    for page_title in search_results[: self.n_pages]
                ]
            )
