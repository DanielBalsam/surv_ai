from core.memory_store.interfaces import Memory
from ..interfaces import ToolInterface


class QueryToolInterface(ToolInterface):
    async def use(self, query: str, *args) -> list[Memory]:
        ...
