from mock import Mock, patch

from surv_ai import ToolResult, TwitterTool
from tests.utils import AsyncMock


async def test_can_use_tool():
    with patch("surv_ai.lib.tools.query.twitter.sntwitter") as mock_twitter_scraper:
        mock_tool = TwitterTool()
        mock_twitter_scraper.TwitterSearchScraper.return_value.get_items.return_value = [
            Mock(
                url="twiter.com/tweet",
                user=Mock(displayname="Jerry"),
                rawContent="Some pithy remark",
            )
        ]
        mock_tool._get_page_text = AsyncMock(return_value=["test"])

        return_val = await mock_tool.use("query")

        assert return_val == [
            ToolResult(
                site_name="Twitter",
                body="Some pithy remark",
                title="Tweet from user named Jerry",
                url="twiter.com/tweet",
            )
        ]
