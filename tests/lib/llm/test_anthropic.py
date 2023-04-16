from mock import Mock, patch

from surv_ai import AnthropicClient, Prompt, PromptMessage
from tests.utils import AsyncMock


async def test_can_get_completion_happy_path():
    with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.raise_for_status = Mock()
        mock_post.return_value.json = AsyncMock(return_value={"completion": "Hello World"})
        gpt_client = AnthropicClient(api_key="123")
        completions = await gpt_client.get_completions(
            [Prompt(messages=[PromptMessage(content="Hello World", role="user", name="User")])]
        )
        assert completions == ["Hello World"]


async def test_can_get_completion_with_multiple_messages():
    with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.raise_for_status = Mock()
        mock_post.return_value.json = AsyncMock(return_value={"completion": "Hello World"})
        gpt_client = AnthropicClient(api_key="123")
        completions = await gpt_client.get_completions(
            [
                Prompt(
                    messages=[
                        PromptMessage(content="Hello World", role="user", name="User"),
                        PromptMessage(content="Hello World", role="user", name="User"),
                    ]
                )
            ]
        )
        assert completions == ["Hello World"]


async def test_can_set_hyper_parameters():
    with patch("aiohttp.ClientSession.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.raise_for_status = Mock()
        mock_post.return_value.json = AsyncMock(return_value={"completion": "Hello World"})
        gpt_client = AnthropicClient(api_key="123")
        await gpt_client.get_completions(
            [
                Prompt(
                    messages=[
                        PromptMessage(content="Hello World", role="user", name="User"),
                        PromptMessage(content="Hello World", role="user", name="User"),
                    ]
                )
            ],
            temperature=0.5,
            top_p=0.5,
            max_tokens=100,
            model="claude-v1",
        )

        assert mock_post.call_args[1]["json"]["temperature"] == 0.5
        assert mock_post.call_args[1]["json"]["top_p"] == 0.5
        assert mock_post.call_args[1]["json"]["max_tokens_to_sample"] == 100
        assert mock_post.call_args[1]["json"]["model"] == "claude-v1"
