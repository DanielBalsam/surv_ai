from pandas import DataFrame

from surv_ai import DataframeTool, ToolResult


async def test_can_use_tool():
    df = DataFrame(
        [
            {
                "title": "Record from a pandas df",
                "content": "Some value",
                "id": "abcd",
            }
        ]
    )
    tool = DataframeTool(
        df,
        "parsed CSV",
        "id",
        "title",
        "content",
    )
    return_val = await tool.use("")

    assert return_val == [
        ToolResult(
            site_name="parsed CSV",
            body="Some value",
            title="Record from a pandas df",
            url="abcd",
        )
    ]
