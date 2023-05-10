import asyncio

from core.assembly.interfaces import AssemblyInterface
from lib.llm.interfaces import LargeLanguageModelClientInterface
from .interfaces import ModelerInterface, DataPoint, Parameter


class Modeler(ModelerInterface):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        assembly_class: type[AssemblyInterface],
        parameters: list[Parameter],
        max_concurrency: int = 1,
    ):
        self.client = client
        self.assembly_class = assembly_class
        self.max_concurrency = max_concurrency
        self.parameters = parameters

    async def model(self, prompt: str) -> list[DataPoint]:
        assembly_index = 0
        results = []

        while assembly_index < len(self.parameters):
            coroutines = []

            for _ in range(self.max_concurrency):
                assembly = self.assembly_class(
                    self.client, **self.parameters[assembly_index].parameters
                )
                coroutines.append(assembly.run(prompt))
                assembly_index += 1

            results += await asyncio.gather(*coroutines)

        return [
            DataPoint(parameter=parameter, response=response)
            for parameter, response in zip(self.parameters, results)
        ]

    @staticmethod
    def get_plot_variables(data_points: list[DataPoint]):
        return (
            [p.parameter.independent_variable for p in data_points],
            [p.response.percent_in_favor for p in data_points],
        )
