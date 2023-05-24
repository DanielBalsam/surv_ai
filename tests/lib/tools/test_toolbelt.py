from surv_ai import Knowledge, ToolBelt
from tests.utils import AsyncMock


async def test_inspect_tool_belt():
    mock_client = AsyncMock()
    mock_client.get_completions = AsyncMock(return_value=["TOOL(hello)"])
    mock_tool = AsyncMock(instruction="TOOL(keywords) - instruction", command=r"TOOL\((.+)\)")
    mock_tool.use = AsyncMock(
        return_value=[
            Knowledge(
                text="Hello World",
                source="test",
            )
        ]
    )
    tool_belt = ToolBelt(
        tools=[mock_tool],
    )

    return_val = await tool_belt.inspect(
        mock_client,
        "prompt",
        [],
    )

    assert tool_belt.tools_as_list(tool_belt.tools) == "1. TOOL(keywords) - instruction"
    assert mock_client.get_completions.call_count == 1
    assert mock_tool.use.call_count == 1
    assert mock_tool.use.call_args[0] == (
        mock_client,
        "prompt",
        "hello",
    )
    assert return_val == [
        Knowledge(
            text="Hello World",
            source="test",
        )
    ]
