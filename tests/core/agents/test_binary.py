from surv_ai import BinaryAgent, Conversation, Knowledge
from tests.utils import AsyncMock


async def test_prompt():
    mock_client = AsyncMock()
    agent = BinaryAgent(mock_client)

    agent.teach_knowledge(
        Knowledge(
            text="This thing is true",
            source="Assertion",
        )
    )

    conversation = Conversation()
    conversation.add("This thing is probably false", "Researcher", "red")
    mock_client.get_completions = AsyncMock(
        return_value=[
            "False",
        ]
    )

    response = await agent.prompt(conversation)

    assert response == "False"
