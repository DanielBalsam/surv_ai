import asyncio
import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

from surv_ai.lib.llm.interfaces import LargeLanguageModelClientInterface
from surv_ai.lib.log import logger

from ..interfaces import ToolInterface, ToolResult


class WikipediaTool(ToolInterface):
    instruction = """
        `WIKIPEDIA(searchTerms)` - use a no more than three keywords to search Wikipedia for additional information.
    """
    command = r"WIKIPEDIA\((.+)\)"

    _base_url = "https://en.wikipedia.org/w/api.php"

    def __init__(
        self,
        llm_client: Optional[LargeLanguageModelClientInterface] = None,
        n_articles=1,
    ):
        if llm_client:
            logger.log_warning(
                "Deprecation warning: LargeLanguageModelClient no longer should be passed into tools on init."
            )

        self.n_articles = n_articles

        self._already_searched = dict()

    async def _search(self, query: str) -> list[str]:
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": re.sub(r"[^A-Za-z0-9 ]+", "", query),
        }
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.get(self._base_url, params=params))
        data = response.json()
        return [result["title"] for result in data["query"]["search"]]

    async def _get_page_text(self, page_title: str) -> list[str]:
        if page_title in self._already_searched:
            return self._already_searched[page_title]

        params = {
            "action": "parse",
            "format": "json",
            "page": page_title,
            "prop": "text",
        }

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.get(self._base_url, params=params))
        data = response.json()

        try:
            html_content = data["parse"]["text"]["*"]

            soup = BeautifulSoup(html_content, "html.parser")

            results = []
            for paragraph in soup.find_all("p"):
                paragraph_text = paragraph.text.strip()

                if paragraph_text:
                    results.append(paragraph_text)

            self._already_searched[page_title] = results
        except Exception:
            return ""

        return results

    async def _ingest_page(
        self,
        page_title: str,
    ):
        logger.log_context(f"......Fetching Wikipedia page with title {page_title}......")
        paragraphs = await self._get_page_text(page_title)
        page_body = "\n\n".join(paragraphs)
        source = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"

        return ToolResult(
            title=f"Wikipedia page titled \"{page_title}\"",
            url=source,
            site_name="Wikipedia",
            body=page_body,
        )

    async def use(
        self,
        search_query: str,
    ) -> list[ToolResult]:
        search_results = await self._search(search_query)

        if len(search_results) == 0:
            return []

        return await asyncio.gather(
            *[self._ingest_page(page_title) for page_title in search_results[: self.n_articles]]
        )
