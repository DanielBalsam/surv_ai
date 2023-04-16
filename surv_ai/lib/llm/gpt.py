import asyncio
from enum import Enum

from aiohttp import ClientSession

from surv_ai.lib.log import logger

from .interfaces import LargeLanguageModelClientInterface, Prompt


class GPTModel(str, Enum):
    TURBO = "gpt-3.5-turbo"
    GPT4 = "gpt-4"


MODEL_TOKEN_LIMITS = {GPTModel.TURBO: 4096, GPTModel.GPT4: 32000}


class GPTClient(LargeLanguageModelClientInterface):
    def __init__(
        self,
        api_key: str,
    ):
        self.api_key = api_key

    async def _get_completion(
        self,
        session: ClientSession,
        prompt: Prompt,
        attempt=1,
        presence_penalty=0,
        frequency_penalty=0,
        temperature=1,
        top_p=1,
        max_tokens: int = 800,
        model=GPTModel.TURBO,
        token_multiplier=1.6,
    ) -> str:
        MAX_PROMPT_TOKENS = MODEL_TOKEN_LIMITS.get(model) - max_tokens
        messages = [
            {
                "role": message.role,
                "content": " ".join(message.content.split()),
            }
            for message in prompt.messages
        ]
        approximate_tokens = len(str(messages).split(" ")) * token_multiplier

        while approximate_tokens > MAX_PROMPT_TOKENS:
            if len(messages) == 1:
                raise Exception("Initial prompt is too long.")

            index_to_remove = None
            for index, _ in enumerate(messages):
                if index == 0:
                    continue

                index_to_remove = index
                break

            if index_to_remove:
                messages.pop(index_to_remove)

            approximate_tokens = len(str(messages).split(" ")) * 1.7

        response = None
        try:
            request = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "max_tokens": max_tokens,
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
            logger.log_exception(e)
            if not response or response.status == 429 or response.status == 502:
                await asyncio.sleep(0.5)

                if attempt < 5:
                    return await self._get_completion(session, prompt, attempt + 1)
            elif response.status == 400:
                if attempt < 5:
                    return await self._get_completion(
                        session,
                        prompt,
                        attempt + 1,
                        token_multiplier=token_multiplier - 0.2,
                    )
            else:
                raise Exception(
                    f"Call to GPT API failed with status {response.status}.",
                    response_body,
                )

        return response_body["choices"][0]["message"]["content"]

    async def get_completions(
        self,
        prompts: list[Prompt],
        presence_penalty=0,
        frequency_penalty=0,
        temperature=1,
        top_p=1,
        max_tokens: int = 800,
        model=GPTModel.TURBO,
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
                        max_tokens=max_tokens,
                        model=model,
                    )
                    for prompt in prompts
                ],
            )
