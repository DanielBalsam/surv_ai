from surv_ai import ReasoningAgent
from tests.utils import AsyncMock


async def test_prompt():
    mock_client = AsyncMock()
    agent = ReasoningAgent(mock_client)

    mock_client.get_completions = AsyncMock(
        return_value=[
            "I think it's true",
        ]
    )

    response = await agent.prompt("test prompt")

    assert response == "I think it's true"
