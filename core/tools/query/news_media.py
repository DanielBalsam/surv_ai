import asyncio
import os
import re

from aiohttp import ClientSession
from bs4 import BeautifulSoup

from core.knowledge_store.interfaces import Knowledge
from lib.log import logger
from lib.llm.interfaces import (
    LargeLanguageModelClientInterface,
    Prompt,
    PromptMessage,
)

from .interfaces import QueryToolInterface


class NewsMediaTool(QueryToolInterface):
    instruction = """
        `NEWS(query)` - use keywords to search the Google News for additional information.
    """
    command = r"NEWS\((.+)\)"

    _base_url = "https://www.googleapis.com/customsearch/v1/siterestrict"

    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        start_date=None,
        end_date=None,
        n_articles=1,
        max_percent_per_source=0.35,
    ):
        self.client = client
        self.n_articles = n_articles

        self.start_date = start_date
        self.end_date = end_date

        self.max_percent_per_source = max_percent_per_source

    async def _search(self, session, query: str) -> list[str]:
        start = 1
        results = []

        seen_sources = {}

        while len(results) < self.n_articles:
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

            try:
                new_records = data["items"]
            except Exception:
                raise Exception("Call to Google API failed.")

            if not new_records:
                break

            if self.max_percent_per_source:
                records_to_return = []

                for record in new_records:
                    if not seen_sources.get(record["displayLink"]):
                        seen_sources[record["displayLink"]] = 0

                    if seen_sources[record["displayLink"]] >= (
                        self.n_articles * self.max_percent_per_source
                    ):
                        continue

                    seen_sources[record["displayLink"]] += 1
                    records_to_return.append(record)

                new_records = records_to_return

            results += new_records
            start += 10

        return results

    async def _get_page_text(
        self, session: ClientSession, web_url: str
    ) -> str:
        response = await session.get(
            web_url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        data = await response.text()

        soup = BeautifulSoup(data, "html.parser")

        results = []
        for paragraph in soup.find_all("p"):
            results.append(paragraph.text)

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
        result: dict,
    ):
        metatags = result["pagemap"]["metatags"][0]

        publication = metatags.get("og:site_name", result["displayLink"])
        title = metatags.get("og:title", result["title"])

        logger.log_context(
            f"......Retrieving {publication} article with title {title}......"
        )
        page_summary = await self._ingest_page_information(
            session,
            original_prompt,
            title,
            publication,
            result["link"],
        )

        return Knowledge(
            text=f'{publication} article entitled "{title}": {page_summary}',
            source=result["link"],
        )

    async def use(
        self,
        original_prompt: str,
        search_query: str,
    ) -> list[Knowledge]:
        async with ClientSession() as session:
            search_results = await self._search(session, search_query)

            if len(search_results) == 0:
                return []

            return await asyncio.gather(
                *[
                    self._ingest_pages(session, original_prompt, web_url)
                    for web_url in search_results[: self.n_articles]
                ]
            )
