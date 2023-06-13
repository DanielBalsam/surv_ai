from surv_ai.lib.llm.interfaces import Prompt, PromptMessage

from ..agent import BaseAgent


class WebPageSummaryAgent(BaseAgent):
    async def _get_prompt(
        self,
        original_prompt: str,
        site_name: str,
        page_title: str,
        page_body: str,
    ):
        return Prompt(
            messages=[
                PromptMessage(
                    content=f"""A user has presented you with a hypothesis:
                    
                    {original_prompt}

                    All subsequent messages will be paragraphs from a {site_name} page titled "{page_title}."

                    Your job is to extract any useful information that might help someone evaluate this hypothesis.

                    Remember to include as much data and concrete examples as possible from the page.

                    For each useful piece of information you extract, please state why it relates to the original hypothesis.
                    """,
                    role="system",
                ),
                *[
                    PromptMessage(content=paragraph, role="user", name="Article")
                    for paragraph in page_body.split("\n\n")
                ],
                PromptMessage(
                    role="assistant",
                    content=f"I have read the {site_name} page entitled {page_title} and some useful information is:",
                ),
            ],
        )

    async def prompt(self, original_prompt: str, site_name: str, page_title: str, page_body: str) -> str:
        prompt = await self._get_prompt(original_prompt, site_name, page_title, page_body)

        response = (await self.client.get_completions([prompt], **self._hyperparameters))[0]

        return response
