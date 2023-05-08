from enum import IntEnum

from colorama import Fore, Style


class AgentLogLevel(IntEnum):
    INFO = 1
    SPEECH = 2
    THOUGHT = 3


class Logger:
    def __init__(self, log_level=AgentLogLevel.SPEECH):
        self.log_level = log_level

    def set_log_level(self, log_level):
        self.log_level = log_level

    def info(self, message: str, color: str = Fore.WHITE):
        if self.log_level >= AgentLogLevel.INFO:
            print(f"{Style.BRIGHT}{color}{message}")

    def speech(self, message: str, color: str = Fore.WHITE):
        if self.log_level >= AgentLogLevel.SPEECH:
            print(f"{Style.BRIGHT}{color}{message}")

    def thought(self, message: str, color: str = Fore.LIGHTBLACK_EX):
        if self.log_level >= AgentLogLevel.THOUGHT:
            print(f"{color}{Style.DIM}{message}")


agent_log = Logger()
