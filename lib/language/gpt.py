import asyncio

from nltk.tokenize import wordpunct_tokenize

from enum import StrEnum
from aiohttp import ClientSession

from .interfaces import LargeLanguageModelClientInterface, Prompt


class GPTModel(StrEnum):
    TURBO = "gpt-3.5-turbo"


class GPTClient(LargeLanguageModelClientInterface):
    def __init__(
        self,
        api_key: str,
        max_tokens: int = 1500,
        model=GPTModel.TURBO,
    ):
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.model = model

    async def _get_completion(
        self,
        session: ClientSession,
        prompt: Prompt,
        attempt=1,
        presence_penalty=0,
        temperature=1,
    ) -> str:
        try:
            request = {
                "model": self.model,
                "messages": [message.dict() for message in prompt.messages],
                "temperature": temperature,
                "presence_penalty": presence_penalty,
                "max_tokens": self.max_tokens,
            }

            response = await session.post(
                "https://api.openai.com/v1/chat/completions",
                json=request,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )

            response.raise_for_status()

            response_body = await response.json()
        except Exception as e:
            if attempt < 5:
                if e.status == 400:
                    prompt.messages = prompt.messages[1:]
                return await self._get_completion(session, prompt, attempt + 1)
            else:
                raise e

        return response_body["choices"][0]["message"]["content"]

    async def get_completions(
        self, prompts: list[Prompt], presence_penalty=0, temperature=1
    ) -> list[str]:
        async with ClientSession() as session:
            return await asyncio.gather(
                *[
                    self._get_completion(
                        session,
                        prompt,
                        presence_penalty=presence_penalty,
                        temperature=temperature,
                    )
                    for prompt in prompts
                ],
            )
