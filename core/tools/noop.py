from core.memory_store.interfaces import MemoryStoreInterface

from lib.language.interfaces import LargeLanguageModelClientInterface
from .interfaces import ToolInterface


class NoopTool(ToolInterface):
    instruction = """
        `CONTINUE()` - you may use this command if you have concluded your research.
    """
    command = r"CONTINUE\(\)"

    def __init__(self, client: LargeLanguageModelClientInterface):
        ...

    async def use(self, query: str, memory_store: MemoryStoreInterface):
        ...
