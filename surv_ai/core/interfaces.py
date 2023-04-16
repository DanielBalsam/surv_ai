from typing import Any, Optional, Protocol

from pydantic import BaseModel
from typing_extensions import TypedDict, Unpack

from surv_ai.lib.knowledge_store.interfaces import Knowledge, KnowledgeStoreInterface
from surv_ai.lib.llm.interfaces import LargeLanguageModelClientInterface
from surv_ai.lib.tools.interfaces import ToolBeltInterface


class AgentInterface(Protocol):
    def __init__(
        self,
        client: LargeLanguageModelClientInterface,
        knowledge_store: Optional[KnowledgeStoreInterface],
        tool_belt: Optional[list[ToolBeltInterface]],
    ):
        ...

    async def prompt(self, input: str) -> str:
        ...

    def teach_text(self, input: str, source: Optional[str] = "User"):
        ...

    def teach_knowledge(self, knowledge: Knowledge):
        ...


class SurveyResponse(BaseModel):
    in_favor: int
    against: int
    undecided: int
    error: int

    percent_in_favor: float
    uncertainty: float


class SurveyKwargs(TypedDict):
    client: LargeLanguageModelClientInterface
    tool_belt: ToolBeltInterface
    n_agents: int
    max_concurrency: int
    max_knowledge_per_agent: int
    base_knowledge: Optional[list[Knowledge]]


class SurveyInterface(Protocol):
    def __init__(self, **kwargs: Unpack[SurveyKwargs]):
        ...

    async def conduct(self, hypothesis: str) -> SurveyResponse:
        ...


class SurveyParameter(BaseModel):
    kwargs: dict
    independent_variable: Any


class DataPoint(BaseModel):
    response: SurveyResponse
    parameter: SurveyParameter


class ModelInterface(Protocol):
    def __init__(
        self,
        survey_class: type[SurveyInterface],
        parameters: list[SurveyParameter],
        max_concurrency: int = 1,
    ):
        ...

    async def build(self, hypothesis: str, *parameter_set: Unpack[dict]) -> list[DataPoint]:
        ...
