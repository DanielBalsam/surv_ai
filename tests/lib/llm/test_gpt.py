from mock import Mock, patch

from surv_ai import GPTClient, Prompt, PromptMessage


async def test_can_get_completion_happy_path():
    with patch("requests.post") as mock_post:
        mock_post.return_value.raise_for_status = Mock()
        mock_post.return_value.json = Mock(return_value={"choices": [{"message": {"content": "Hello World"}}]})
        gpt_client = GPTClient(api_key="123")
        completions = await gpt_client.get_completions(
            [Prompt(messages=[PromptMessage(content="Hello World", role="user", name="User")])]
        )
        assert completions == ["Hello World"]


async def test_can_get_completion_with_multiple_messages():
    with patch("requests.post") as mock_post:
        mock_post.return_value.raise_for_status = Mock()
        mock_post.return_value.json = Mock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": "Hello World",
                            "role": "bot",
                            "name": "Bot",
                        }
                    },
                ]
            }
        )
        gpt_client = GPTClient(api_key="123")
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
    with patch("requests.post") as mock_post:
        mock_post.return_value.raise_for_status = Mock()
        mock_post.return_value.json = Mock(return_value={"choices": [{"message": {"content": "Hello World"}}]})
        gpt_client = GPTClient(api_key="123")
        await gpt_client.get_completions(
            [Prompt(messages=[PromptMessage(content="Hello World", role="user", name="User")])],
            presence_penalty=0.5,
            frequency_penalty=0.5,
            temperature=0.5,
            top_p=0.5,
            max_tokens=100,
            model="gpt-4",
        )

        assert mock_post.call_args[1]["json"]["presence_penalty"] == 0.5
        assert mock_post.call_args[1]["json"]["frequency_penalty"] == 0.5
        assert mock_post.call_args[1]["json"]["temperature"] == 0.5
        assert mock_post.call_args[1]["json"]["top_p"] == 0.5
        assert mock_post.call_args[1]["json"]["max_tokens"] == 100
        assert mock_post.call_args[1]["json"]["model"] == "gpt-4"
