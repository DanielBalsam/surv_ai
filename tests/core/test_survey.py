from mock import patch

from surv_ai import Survey, ToolResult
from tests.utils import AsyncMock


async def test_conduct():
    with patch("surv_ai.core.survey.ReasoningAgent") as mock_reasoning_agent, patch(
        "surv_ai.core.survey.BinaryAgent"
    ) as mock_binary_agent, patch("surv_ai.core.survey.WebPageSummaryAgent") as mock_summary_agent:
        mock_reasoning_agent.return_value.color = "red"
        mock_binary_agent.return_value.color = "blue"

        mock_reasoning_agent.return_value.prompt = AsyncMock(return_value="I think it's true")
        mock_binary_agent.return_value.prompt = AsyncMock(return_value="True")
        mock_summary_agent.return_value.prompt = AsyncMock(return_value="Test page summary: its a test")

        mock_tool_belt = AsyncMock()
        mock_client = AsyncMock()
        survey = Survey(
            client=mock_client,
            tool_belt=mock_tool_belt,
            n_agents=10,
        )
        mock_tool_belt.inspect = AsyncMock(
            return_value=[
                ToolResult(
                    url="test",
                    body="test",
                    title="test",
                    site_name="test",
                )
            ]
        )

        response = await survey.conduct("test prompt")

        assert survey.tool_belt.inspect.call_count == 1
        assert survey.tool_belt.inspect.call_args_list[0][0] == (mock_client, "test prompt", [])

        assert mock_reasoning_agent.call_count == 10
        assert mock_binary_agent.call_count == 10

        assert mock_reasoning_agent.return_value.prompt.call_count == 10
        assert mock_binary_agent.return_value.prompt.call_count == 10

        assert response.percent_in_favor == 1.0
