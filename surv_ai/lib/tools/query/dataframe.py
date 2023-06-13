from surv_ai.lib.tools.interfaces import ToolResult

from ..interfaces import ToolInterface


class DataframeTool(ToolInterface):
    instruction = """
        `DATAFRAME()` - Load relevant information from a Pandas dataframe.
    """
    command = r"DATAFRAME\((.+)\)"

    def __init__(
        self,
        dataframe,
        source_of_data: str,
        id_column_name: str,
        title_column_column: str,
        content_column_name: str,
    ):
        self.dataframe = dataframe
        self.source_of_data = source_of_data
        self.id_column_name = id_column_name
        self.title_column_column = title_column_column
        self.content_column_name = content_column_name

    async def use(
        self,
        _,
    ) -> list[ToolResult]:
        return [
            ToolResult(
                url=row[self.id_column_name],
                site_name=self.source_of_data,
                title=row[self.title_column_column],
                body=row[self.content_column_name],
            )
            for _, row in self.dataframe.iterrows()
        ]
