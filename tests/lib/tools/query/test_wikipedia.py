from mock import patch

from surv_ai import Knowledge, WikipediaTool
from tests.utils import AsyncMock


async def test_can_use_tool():
    with patch("requests.post"):
        mock_client = AsyncMock()
        mock_client.get_completions = AsyncMock(return_value=["an article titled hello world"])
        mock_tool = WikipediaTool()
        mock_tool._search = AsyncMock(
            return_value=[
                "Hello World",
            ]
        )

        return_val = await mock_tool.use(
            mock_client,
            "prompt",
            [],
        )

        assert return_val == [
            Knowledge(
                text="Wikipedia page entitled an article titled hello world: an article titled hello world",
                source="https://en.wikipedia.org/wiki/an_article_titled_hello_world",
            )
        ]
