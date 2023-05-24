import asyncio
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

from surv_ai.lib.knowledge_store.interfaces import Knowledge
from surv_ai.lib.llm.interfaces import (
    LargeLanguageModelClientInterface,
    Prompt,
    PromptMessage,
)
from surv_ai.lib.log import logger

from .interfaces import QueryToolInterface


class GoogleCustomSearchTool(QueryToolInterface):
    instruction = """
        `SEARCH(keywords)` - use keywords to search the web for additional information.
    """
    command = r"SEARCH\((.+)\)"

    _base_url = "https://www.googleapis.com/customsearch/v1/siterestrict"

    def __init__(
        self,
        llm_client: Optional[LargeLanguageModelClientInterface] = None,
        google_api_key: str = "",
        google_search_engine_id: str = "",
        start_date=None,
        end_date=None,
        n_pages=1,
        max_percent_per_source=1.0,
        max_concurrency=10,
        only_include_sources=None,
    ):
        if llm_client:
            logger.log_warning(
                "Deprecation warning: LargeLanguageModelClient no longer should be passed into tools on init."
            )

        if not google_api_key:
            raise ValueError("google_api_key is required for GoogleCustomSearchTool")
        elif not google_search_engine_id:
            raise ValueError("google_search_engine_id is required for GoogleCustomSearchTool")

        self.google_api_key = google_api_key
        self.google_search_engine_id = google_search_engine_id

        self.n_pages = n_pages

        self.start_date = start_date
        self.end_date = end_date

        self.max_percent_per_source = max_percent_per_source
        self.max_concurrency = max_concurrency

        self.only_include_sources = only_include_sources

    async def _search(self, query: str) -> list[str]:
        start = 1
        results = []

        seen_sources = {}

        while len(results) < self.n_pages:
            params = {
                "key": self.google_api_key,
                "cx": self.google_search_engine_id,
                "q": re.sub(r"[^A-Za-z0-9 ]+", "", query),
                "start": start,
                "num": 10,
            }

            if self.start_date:
                params["q"] += f" after:{self.start_date}"

            if self.end_date:
                params["q"] += f" before:{self.end_date}"

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: requests.get(self._base_url, params=params))
            data = response.json()

            try:
                new_records = data["items"]
            except Exception:
                logger.log_exception("Could not retrieve all articles.")

                return results

            if not new_records:
                break

            if self.max_percent_per_source:
                records_to_return = []

                for record in new_records:
                    if not seen_sources.get(record["displayLink"]):
                        seen_sources[record["displayLink"]] = 0

                    if self.only_include_sources:
                        compatible_source = False
                        for source in self.only_include_sources:
                            if source in record["displayLink"]:
                                compatible_source = True

                        if not compatible_source:
                            continue

                    if seen_sources[record["displayLink"]] >= (self.n_pages * self.max_percent_per_source):
                        continue

                    seen_sources[record["displayLink"]] += 1
                    records_to_return.append(record)

                new_records = records_to_return

            results += new_records
            start += 10

        return results

    async def _get_page_text(self, web_url: str) -> str:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(web_url, headers={"User-Agent": "Mozilla/5.0"}),
        )

        data = response.text

        soup = BeautifulSoup(data, "html.parser")

        results = []
        for paragraph in soup.find_all("p"):
            results.append(paragraph.text)

        return results

    async def _ingest_page_information(
        self,
        llm_client: LargeLanguageModelClientInterface,
        original_prompt: str,
        title: str,
        publication: str,
        web_url: str,
    ):
        paragraphs = await self._get_page_text(web_url)

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
                *[PromptMessage(content=paragraph, role="user", name="Article") for paragraph in paragraphs],
                PromptMessage(
                    role="assistant",
                    content=f"I have read the Wikpedia article entitled {title} and some useful information is:",
                ),
            ],
        )
        response = await llm_client.get_completions([prompt])

        return response[0]

    async def _ingest_pages(
        self,
        llm_client: LargeLanguageModelClientInterface,
        original_prompt: str,
        result: dict,
    ):
        try:
            metatags = result["pagemap"]["metatags"][0]

            publication = metatags.get("og:site_name", result["displayLink"])
            title = metatags.get("og:title", result["title"])

            logger.log_context(f"......Retrieving {publication} article with title {title}......")
            page_summary = await self._ingest_page_information(
                llm_client,
                original_prompt,
                title,
                publication,
                result["link"],
            )
            logger.log_internal(f"Summary of  {publication} article with title {title}: {page_summary}")

            return Knowledge(
                text=f'{publication} article entitled "{title}": {page_summary}',
                source=result["link"],
            )
        except Exception as e:
            logger.log_exception(e)
            return None

    async def use(
        self,
        llm_client: LargeLanguageModelClientInterface,
        original_prompt: str,
        search_query: str,
    ) -> list[Knowledge]:
        search_results = await self._search(search_query)

        if len(search_results) == 0:
            return []

        response = []

        while len(response) < len(search_results[: self.n_pages]):
            results_to_fetch = min(
                self.max_concurrency,
                len(search_results[: self.n_pages]) - len(response),
            )

            response += await asyncio.gather(
                *[
                    self._ingest_pages(llm_client, original_prompt, result)
                    for result in search_results[len(response) : len(response) + results_to_fetch]
                ]
            )

        return [r for r in response if r]
