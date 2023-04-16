from surv_ai import Knowledge, LocalKnowledgeStore


def test_can_add_text():
    knowledge_store = LocalKnowledgeStore()
    knowledge_store.add_text("Hello World", "User")
    assert knowledge_store.knowledge == [Knowledge(text="Hello World", source="User")]


def test_can_add_knowledge():
    knowledge_store = LocalKnowledgeStore()
    knowledge_store.add_knowledge(Knowledge(text="Hello World", source="User"))
    assert knowledge_store.knowledge == [Knowledge(text="Hello World", source="User")]


def test_can_filter_knowledge():
    knowledge_store = LocalKnowledgeStore()
    knowledge_store.add_knowledge(Knowledge(text="Hello World", source="User"))
    knowledge_store.add_knowledge(Knowledge(text="Hello World", source="System"))
    knowledge_store.add_knowledge(Knowledge(text="Hello World", source="User"))
    assert (
        knowledge_store.filter_knowledge(include_sources=["User"])
        == [
            Knowledge(text="Hello World", source="User"),
            Knowledge(text="Hello World", source="User"),
        ]
        == knowledge_store.filter_knowledge(exclude_sources=["System"])
    )


def test_can_recall_recent():
    knowledge_store = LocalKnowledgeStore()
    knowledge_store.add_knowledge(Knowledge(text="Hello World", source="User"))
    knowledge_store.add_knowledge(Knowledge(text="Hello World", source="System"))
    knowledge_store.add_knowledge(Knowledge(text="Hello World", source="User"))
    assert (
        knowledge_store.recall_recent(n_knowledge_items=1)
        == [Knowledge(text="Hello World", source="User")]
        == knowledge_store.recall_recent(n_knowledge_items=1, include_sources=["User"])
        == knowledge_store.recall_recent(n_knowledge_items=1, exclude_sources=["System"])
    )


def test_can_serialize_knowledge():
    knowledge_store = LocalKnowledgeStore()
    knowledge_store.add_knowledge(Knowledge(text="Hello World 1", source="User"))
    knowledge_store.add_knowledge(Knowledge(text="Hello World 2", source="System"))
    knowledge_store.add_knowledge(Knowledge(text="Hello World 3", source="User"))
    assert (
        knowledge_store.knowledge_as_string(knowledge_store.knowledge)
        == """1. Hello World 1
2. Hello World 2
3. Hello World 3"""
    )
