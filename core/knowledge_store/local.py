from typing import Optional

from .interfaces import Knowledge, KnowledgeStoreInterface


class LocalKnowledgeStore(KnowledgeStoreInterface):
    def __init__(self):
        self.knowledge: list[Knowledge] = []

    def add_text(self, input: str, source: Optional[str] = None):
        self.knowledge.append(Knowledge(text=input, source=source))

    def add_knowledge(self, knowledge: Knowledge):
        self.knowledge.append(knowledge)

    def filter_knowledge(
        self,
        include_sources: Optional[list[str]] = None,
        exclude_sources: Optional[list[str]] = None,
    ):
        knowledge = self.knowledge

        if include_sources:
            knowledge = [
                knowledge_item
                for knowledge_item in knowledge
                if knowledge_item.source in include_sources
            ]

        if exclude_sources:
            knowledge = [
                knowledge_item
                for knowledge_item in knowledge
                if knowledge_item.source not in exclude_sources
            ]

        return knowledge

    def recall_recent(
        self,
        n_knowledge_items=5,
        include_sources: Optional[list[str]] = None,
        exclude_sources: Optional[list[str]] = None,
    ) -> list[Knowledge]:
        return [
            knowledge_item
            for knowledge_item in self.filter_knowledge(
                include_sources, exclude_sources
            )
        ][-n_knowledge_items:]

    @staticmethod
    def knowledge_as_string(knowledge: list[Knowledge]) -> str:
        return (
            "\n".join(f"{i + 1}. {m.text}" for i, m in enumerate(knowledge))
            if knowledge
            else ""
        )
