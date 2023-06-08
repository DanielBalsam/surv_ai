from surv_ai import WebPageSummaryAgent
from tests.utils import AsyncMock


async def test_prompt():
    mock_client = AsyncMock()
    agent = WebPageSummaryAgent(mock_client)

    mock_client.get_completions = AsyncMock(
        return_value=[
            "Summary of a page",
        ]
    )

    response = await agent.prompt("test prompt", "google.com", "Test Page", "a page body to be summarized")

    assert response == "Summary of a page"
