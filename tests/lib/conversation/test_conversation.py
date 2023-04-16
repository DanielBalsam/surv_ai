from surv_ai import Conversation


def test_can_add():
    convo = Conversation()
    convo.add("Hello", "Bob")
    assert len(convo) == 1
    assert convo[0].text == "Hello"
    assert convo[0].speaker == "Bob"


def test_can_iterate():
    convo = Conversation()
    convo.add("Hello", "Bob")
    convo.add("Hi", "Alice")
    assert [message.text for message in convo] == ["Hello", "Hi"]


def test_can_serialize_as_string():
    convo = Conversation()
    convo.add("Hello", "Bob")
    convo.add("Hi", "Alice")
    assert (
        convo.as_string()
        == """```
Bob said, "Hello"

Alice said, "Hi"
```"""
    )
