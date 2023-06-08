from mock import Mock, patch

from surv_ai import GoogleCustomSearchTool, ToolResult
from tests.utils import AsyncMock


async def test_can_use_tool():
    with patch("requests.get", new_callable=Mock) as mock_get:
        mock_get.return_value.text = "<p>test</p>"
        mock_tool = GoogleCustomSearchTool(
            google_api_key="123",
            google_search_engine_id="456",
        )
        mock_tool._search = AsyncMock(
            return_value=[
                {
                    "title": "Hello World",
                    "link": "https://www.google.com",
                    "displayLink": "https://www.google.com",
                    "pagemap": {
                        "metatags": [
                            {
                                "og:description": "Hello World",
                            }
                        ]
                    },
                }
            ]
        )

        return_val = await mock_tool.use("query val")

        assert mock_tool._search.call_count == 1
        assert return_val == [
            ToolResult(
                body="test",
                site_name="https://www.google.com",
                title="Hello World",
                url="https://www.google.com",
            )
        ]
