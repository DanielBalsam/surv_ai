import snscrape.modules.twitter as sntwitter

from surv_ai.lib.tools.interfaces import ToolResult

from ..interfaces import ToolInterface


class TwitterTool(ToolInterface):
    instruction = """
        `TWITTER(subject)` - query Twitter for sentiment on a specific subject. Use as simple a query as possible. For instance, if the query is "Is Joe Biden's economic policy popular?" you might query "TWITTER(Joe Biden economy)".
    """
    command = r"TWITTER\((.+)\)"

    def __init__(
        self,
        start_date=None,
        end_date=None,
        n_tweets=10,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.n_tweets = n_tweets

    async def use(
        self,
        search_query: str,
    ) -> list[ToolResult]:
        results = []

        if self.start_date:
            search_query += " since:" + self.start_date
        if self.end_date:
            search_query += " until:" + self.end_date

        query = sntwitter.TwitterSearchScraper(search_query, mode=sntwitter.TwitterSearchScraperMode.TOP)

        for i, tweet in enumerate(query.get_items()):
            if i >= self.n_tweets:
                break

            results.append(
                ToolResult(
                    url=tweet.url,
                    site_name="Twitter",
                    title=f"Tweet from user named {tweet.user.displayname}",
                    body=tweet.rawContent,
                )
            )

        return results
