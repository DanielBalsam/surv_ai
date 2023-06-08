from mock import patch

from surv_ai import ToolResult, WikipediaTool
from tests.utils import AsyncMock


async def test_can_use_tool():
    with patch("requests.post"):
        mock_tool = WikipediaTool()
        mock_tool._search = AsyncMock(
            return_value=[
                "Hello World",
            ]
        )
        mock_tool._get_page_text = AsyncMock(return_value=["test"])

        return_val = await mock_tool.use("query")

        assert return_val == [
            ToolResult(
                site_name="Wikipedia",
                body="test",
                title="Hello World",
                url="https://en.wikipedia.org/wiki/Hello_World",
            )
        ]
