from typing import Any, Protocol, Unpack
from pydantic import BaseModel
from core.assembly.interfaces import AssemblyInterface, AssemblyResponse

from lib.llm.interfaces import LargeLanguageModelClientInterface


class Parameter(BaseModel):
    parameters: dict
    independent_variable: Any


class DataPoint(BaseModel):
    response: AssemblyResponse
    parameter: Parameter


class ModelerInterface(Protocol):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        assembly_class: type[AssemblyInterface],
        parameters: list[Parameter],
        max_concurrency: int = 1,
    ):
        ...

    async def run(
        self, prompt: str, *parameter_set: Unpack[dict]
    ) -> list[DataPoint]:
        ...
