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


class NewsMediaTool(QueryToolInterface):
    instruction = """
        `NEWS(query)` - use keywords to search the New York Times for additional information.
    """
    command = r"NEWS\((.+)\)"

    _base_url = "https://www.googleapis.com/customsearch/v1/siterestrict"

    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        embeddings: Optional[EmbeddingInterface] = None,
        start_date=None,
        end_date=None,
        n_pages=1,
    ):
        if not embeddings:
            embeddings = SentenceBertEmbedding()

        self.client = client
        self.n_pages = n_pages
        self.embeddings = embeddings

        self.start_date = start_date
        self.end_date = end_date

        self._already_searched = dict()

    async def _search(self, session, query: str) -> list[str]:
        start = 1
        results = []

        while len(results) < self.n_pages:
            params = {
                "key": os.getenv("GOOGLE_API_KEY", ""),
                "cx": os.getenv("GOOGLE_SEARCH_ENGINE_ID", ""),
                "q": re.sub(r"[^A-Za-z0-9 ]+", "", query),
                "start": start,
                "num": 10,
            }

            if self.start_date:
                params["q"] += f" after:{self.start_date}"

            if self.end_date:
                params["q"] += f" before:{self.end_date}"

            async with session.get(self._base_url, params=params) as response:
                data = await response.json()

            new_records = data["items"]

            if not new_records:
                break

            results += new_records
            start += 10

        return results

    async def _get_page_text(self, _, web_url: str) -> str:
        if web_url in self._already_searched:
            return self._already_searched[web_url]

        data = subprocess.run(
            "curl -s " + web_url + ' --header "User-Agent: Mozilla/5.0"',
            shell=True,
            capture_output=True,
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
        title: str,
        publication: str,
        web_url: str,
    ):
        paragraphs = await self._get_page_text(session, web_url)

        prompt = Prompt(
            messages=[
                PromptMessage(
                    content=f"""A user has asked you some questions:
                    
                    {original_prompt}

                    All subsequent messages will be paragraphs from a {publication} article titled "{title}."

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
                    content=f"I have read the Wikpedia article entitled {title} and some useful information is:",
                ),
            ],
        )
        response = await self.client.get_completions([prompt])

        return response[0]

    async def _ingest_pages(
        self,
        session: ClientSession,
        original_prompt: str,
        relevant_context: str,
        result: dict,
    ):
        metatags = result["pagemap"]["metatags"][0]

        publication = metatags.get("og:site_name", "news")
        title = metatags.get("og:title", result["title"])

        agent_log.thought(
            f"......Retrieving {publication} article with title {title}......"
        )
        page_summary = await self._ingest_page_information(
            session,
            original_prompt,
            title,
            publication,
            result["link"],
        )

        page_embeddings = self.embeddings.embed([page_summary])[0]
        return Memory(
            text=f'{publication} article entitled "{title}": {page_summary}',
            source=result["link"],
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

            return [
                await self._ingest_pages(
                    session, original_prompt, relevant_context, web_url
                )
                for web_url in search_results[: self.n_pages]
            ]
