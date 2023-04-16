from surv_ai.lib.knowledge_store.interfaces import Knowledge

from ..interfaces import ToolInterface


class QueryToolInterface(ToolInterface):
    async def use(self, query: str, *args) -> list[Knowledge]:
        ...
