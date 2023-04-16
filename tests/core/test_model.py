from mock import Mock

from surv_ai import Model, SurveyParameter, SurveyResponse
from tests.utils import AsyncMock


async def test_build():
    mock_survey = Mock()
    mock_survey.return_value = AsyncMock()
    mock_survey.return_value.conduct.return_value = SurveyResponse(
        percent_in_favor=0.5,
        in_favor=1,
        against=1,
        undecided=0,
        uncertainty=0,
        error=0,
    )
    model = Model(
        survey_class=mock_survey,
        parameters=[
            SurveyParameter(
                independent_variable="test",
                kwargs={"test": "test"},
            ),
            SurveyParameter(
                independent_variable="test 2",
                kwargs={"test": "test 2"},
            ),
        ],
    )
    data_points = await model.build("test")
    assert len(data_points) == 2
    assert data_points[0].parameter.independent_variable == "test"
    assert mock_survey.call_args_list[0][1]["test"] == "test"
    assert data_points[0].response.percent_in_favor == 0.5

    assert data_points[1].parameter.independent_variable == "test 2"
    assert mock_survey.call_args_list[1][1]["test"] == "test 2"
    assert data_points[0].response.percent_in_favor == 0.5
