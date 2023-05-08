import asyncio
from enum import StrEnum

from aiohttp import ClientSession

from .interfaces import LargeLanguageModelClientInterface, Prompt


class GPTModel(StrEnum):
    TURBO = "gpt-3.5-turbo"


MODEL_TOKEN_LIMITS = {GPTModel.TURBO: 4096}


class GPTClient(LargeLanguageModelClientInterface):
    def __init__(
        self,
        api_key: str,
        max_tokens: int = 500,
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
        frequency_penalty=0,
        temperature=1,
        top_p=1,
    ) -> str:
        MAX_PROMPT_TOKENS = (
            MODEL_TOKEN_LIMITS.get(self.model) - self.max_tokens
        )
        messages = [
            {
                "role": message.role,
                "content": " ".join(message.content.split()),
            }
            for message in prompt.messages
        ]
        approximate_tokens = len(str(messages).split(" ")) * 1.5

        while approximate_tokens > MAX_PROMPT_TOKENS:
            if len(messages) == 1:
                raise Exception("Initial prompt is too long.")

            index_to_remove = None
            for index, message in enumerate(messages):
                if message["role"] == "system":
                    continue

                index_to_remove = index
                break

            if index_to_remove:
                messages.pop(index_to_remove)

            approximate_tokens = len(str(messages).split(" ")) * 1.6

        try:
            request = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
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

            try:
                response_body = await response.json()
            except Exception:
                response_body = await response.text()

            response.raise_for_status()
        except Exception as e:
            if response.status == 429:
                await asyncio.sleep(0.5)

                if attempt < 5:
                    return await self._get_completion(
                        session, prompt, attempt + 1
                    )
            else:
                raise e

        return response_body["choices"][0]["message"]["content"]

    async def get_completions(
        self,
        prompts: list[Prompt],
        presence_penalty=0,
        frequency_penalty=0,
        temperature=1,
        top_p=1,
    ) -> list[str]:
        async with ClientSession() as session:
            return await asyncio.gather(
                *[
                    self._get_completion(
                        session,
                        prompt,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    for prompt in prompts
                ],
            )
