import asyncio
from enum import Enum

from aiohttp import ClientSession

from surv_ai.lib.log import logger

from .interfaces import LargeLanguageModelClientInterface, Prompt


class AnthropicModel(str, Enum):
    CLAUDE_V1 = "claude-v1"


MODEL_TOKEN_LIMITS = {AnthropicModel.CLAUDE_V1: 100000}


class AnthropicClient(LargeLanguageModelClientInterface):
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
        model=AnthropicModel.CLAUDE_V1,
        token_multiplier=1.6,
    ) -> str:
        for message in prompt.messages:
            if message.role in ["user", "system"]:
                message.role = "Human"
            elif message.role == "assistant":
                message.role = "Assistant"

        messages = (
            "\n\n".join(
                [f"{message.name if message.name else message.role}: {message.content}" for message in prompt.messages]
            )
            + "\n\nAssistant: "
        )

        response = None
        try:
            request = {
                "model": model,
                "prompt": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens_to_sample": max_tokens,
                "stop_sequences": ["\n\nUser:"],
            }

            response = await session.post(
                "https://api.anthropic.com/v1/complete",
                json=request,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": f"{self.api_key}",
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

        return response_body["completion"]

    async def get_completions(
        self,
        prompts: list[Prompt],
        presence_penalty=0,
        frequency_penalty=0,
        temperature=1,
        top_p=1,
        max_tokens: int = 800,
        model=AnthropicModel.CLAUDE_V1,
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
