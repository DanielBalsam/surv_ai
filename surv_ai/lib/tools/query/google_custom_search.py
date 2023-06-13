import asyncio
import re

import requests
from bs4 import BeautifulSoup

from surv_ai.lib.log import logger
from surv_ai.lib.tools.interfaces import ToolResult

from ..interfaces import ToolInterface


class GoogleCustomSearchTool(ToolInterface):
    instruction = """
        `SEARCH(keywords)` - use keywords to search the web for additional information.
    """
    command = r"SEARCH\((.+)\)"

    _base_url = "https://www.googleapis.com/customsearch/v1/siterestrict"

    def __init__(
        self,
        google_api_key: str = "",
        google_search_engine_id: str = "",
        start_date=None,
        end_date=None,
        n_pages=10,
        only_include_websites=None,
    ):
        if not google_api_key:
            raise ValueError("google_api_key is required for GoogleCustomSearchTool")
        elif not google_search_engine_id:
            raise ValueError("google_search_engine_id is required for GoogleCustomSearchTool")

        self.google_api_key = google_api_key
        self.google_search_engine_id = google_search_engine_id
        self.n_pages = n_pages
        self.start_date = start_date
        self.end_date = end_date
        self.only_include_websites = only_include_websites

    async def _search(self, query: str) -> list[str]:
        start = 1
        results = []

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
                logger.log_exception("Could not retrieve all pages.")

                return results

            if not new_records:
                break

            results += new_records
            start += 10

        return results

    def _get_page_text(self, web_url: str) -> str:
        response = requests.get(web_url, headers={"User-Agent": "Mozilla/5.0"})

        data = response.text

        soup = BeautifulSoup(data, "html.parser")

        results = []
        for paragraph in soup.find_all("p"):
            results.append(paragraph.text)

        return "\n\n".join(results)

    def _ingest_page(
        self,
        result: dict,
    ):
        try:
            metatags = result["pagemap"]["metatags"][0]

            site_name = metatags.get("og:site_name", result["displayLink"])
            title = metatags.get("og:title", result["title"])

            logger.log_context(f"......Retrieving {site_name} page with title {title}......")
            page_text = self._get_page_text(result["link"])

            return ToolResult(
                url=result["link"],
                title=f"{site_name} page titled \"{title}\"",
                body=page_text,
                site_name=site_name,
            )
        except Exception as e:
            logger.log_exception(e)
            return None

    async def use(
        self,
        search_query: str,
    ) -> list[ToolResult]:
        search_results = await self._search(search_query)

        if len(search_results) == 0:
            return []

        response = [self._ingest_page(result) for result in search_results[0 : self.n_pages]]

        return [r for r in response if r]
