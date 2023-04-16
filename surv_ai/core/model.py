import asyncio

from .interfaces import DataPoint, ModelInterface, SurveyInterface, SurveyParameter


class Model(ModelInterface):
    def __init__(
        self,
        survey_class: type[SurveyInterface],
        parameters: list[SurveyParameter],
        max_concurrency: int = 1,
    ):
        self.survey_class = survey_class
        self.max_concurrency = max_concurrency
        self.parameters = parameters

    async def build(self, hypothesis: str) -> list[DataPoint]:
        survey_index = 0
        results = []

        while survey_index < len(self.parameters):
            coroutines = []

            for _ in range(self.max_concurrency):
                survey = self.survey_class(**self.parameters[survey_index].kwargs)
                coroutines.append(survey.conduct(hypothesis))
                survey_index += 1

            results += await asyncio.gather(*coroutines)

        return [
            DataPoint(parameter=parameter, response=response) for parameter, response in zip(self.parameters, results)
        ]

    @staticmethod
    def get_plot_variables(data_points: list[DataPoint]):
        return (
            [p.parameter.independent_variable for p in data_points],
            [p.response.percent_in_favor for p in data_points],
        )
