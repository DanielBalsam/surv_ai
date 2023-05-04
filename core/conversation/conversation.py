from typing import Optional
from lib.agent_log import agent_log
from .interfaces import ConversationInterface, ChatMessage


class Conversation(ConversationInterface):
    def __init__(self):
        self.history: list[ChatMessage] = []

    def add(self, message: str, speaker: str, color=Optional[str]):
        agent_log.speech(f"{speaker}: {message}", color)
        self.history.append(
            ChatMessage(text=message, speaker=speaker, color=color)
        )

    def __iter__(self):
        return iter(self.history)

    def __len__(self):
        return len(self.history)

    def __getitem__(self, index):
        return self.history[index]

    def as_string(
        self, n_most_recent=5, exclude_speakers: Optional[list[str]] = None
    ):
        if exclude_speakers is None:
            exclude_speakers = []

        return (
            "```\n"
            + "\n\n".join(
                [
                    str(message)
                    for message in self.history[-n_most_recent:]
                    if message.speaker not in exclude_speakers
                ]
            )
            + "\n```"
        )