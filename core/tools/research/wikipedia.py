from typing import Optional
from bs4 import BeautifulSoup
import re
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
from lib.agent_log import agent_log

from .interfaces import QueryToolInterface


class WikipediaTool(QueryToolInterface):
    instruction = """
        `WIKIPEDIA(searchTerms)` - use a no more than three keywords to search Wikipedia for additional information.
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

        self._already_searched = dict()

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
        if page_title in self._already_searched:
            return self._already_searched[page_title]

        params = {
            "action": "parse",
            "format": "json",
            "page": page_title,
            "prop": "text",
        }

        async with session.get(self._base_url, params=params) as response:
            data = await response.json()

            try:
                html_content = data["parse"]["text"]["*"]

                soup = BeautifulSoup(html_content, "html.parser")

                results = []
                for paragraph in soup.find_all("p"):
                    paragraph_text = paragraph.text.strip()

                    if paragraph_text:
                        results.append(paragraph_text)

                self._already_searched[page_title] = results
            except:
                return ""

            return results

    async def _ingest_page_information(
        self,
        session,
        original_prompt: str,
        page_title: str,
    ):
        paragraphs = await self._get_page_text(session, page_title)

        prompt = Prompt(
            messages=[
                PromptMessage(
                    content=f"""A user has asked you some questions:
                    
                    {original_prompt}

                    All subsequent messages will be paragraphs from a Wikipedia article is entitled "{page_title}."

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
                    content=f"I have read the Wikpedia article entitled {page_title} and some useful information is:",
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
        page_title: str,
    ):
        agent_log.thought(
            f"......Learning Wikipedia page with title {page_title}......"
        )
        page_summary = await self._ingest_page_information(
            session,
            original_prompt,
            page_title,
        )

        source = (
            f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
        )

        page_embeddings = self.embeddings.embed([page_summary])[0]
        return Memory(
            text=f"Wikipedia page entitled {page_title}: {page_summary}",
            source=source,
            embedding=page_embeddings,
        )

    async def _filter_relevant_pages(
        self,
        original_prompt: str,
        relevant_context: str,
        search_results: list[str],
    ):
        prompt = Prompt(
            messages=[
                PromptMessage(
                    content=f"""Your job is to determine which, if any of the following article titles are relevant to a user's prompt.
                                   
                    Original prompt:
                    
                    '{original_prompt}'.
                    
                    Relevant context:

                    {relevant_context}

                    The next message will be the the list of articles separated by commas.

                    Please return only the relevant article titles, also separated by commas.
                    """,
                    role="system",
                ),
                PromptMessage(
                    content=", ".join(search_results),
                    role="user",
                ),
            ],
        )
        response = (await self.client.get_completions([prompt]))[0]

        return response

    async def use(
        self,
        original_prompt: str,
        relevant_context: str,
        search_query: str,
    ) -> list[Memory]:
        async with ClientSession() as session:
            search_results = await self._search(session, search_query)

            relevant_results = (
                await self._filter_relevant_pages(
                    original_prompt, relevant_context, search_results
                )
            ).split(", ")

            if len(relevant_results) == 0:
                return []

            return await asyncio.gather(
                *[
                    self._ingest_pages(session, original_prompt, page_title)
                    for page_title in relevant_results[: self.n_pages]
                ]
            )
