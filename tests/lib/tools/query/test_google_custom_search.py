from mock import Mock, patch

from surv_ai import GoogleCustomSearchTool, Knowledge
from tests.utils import AsyncMock


async def test_can_use_tool():
    with patch("requests.get", new_callable=Mock) as mock_get:
        mock_get.return_value.text = "test"
        mock_client = AsyncMock()
        mock_client.get_completions = AsyncMock(return_value=["an article titled hello world"])
        mock_tool = GoogleCustomSearchTool(
            llm_client=mock_client,
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

        return_val = await mock_tool.use(
            "prompt",
            [],
        )

        assert mock_client.get_completions.call_count == 1
        assert mock_tool._search.call_count == 1
        assert return_val == [
            Knowledge(
                text='https://www.google.com article entitled "Hello World": an article titled hello world',
                source="https://www.google.com",
            )
        ]
