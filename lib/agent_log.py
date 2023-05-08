from enum import IntEnum

from colorama import Fore, Style


class AgentLogLevel(IntEnum):
    CONTEXT = 1
    OUTPUT = 2
    INTERNAL = 3


class Logger:
    def __init__(self, log_level=AgentLogLevel.OUTPUT):
        self.log_level = log_level

    def set_log_level(self, log_level):
        self.log_level = log_level

    def log_context(self, message: str, color: str = Fore.WHITE):
        if self.log_level >= AgentLogLevel.CONTEXT:
            print(f"{Style.BRIGHT}{color}{message}")

    def log_output(self, message: str, color: str = Fore.WHITE):
        if self.log_level >= AgentLogLevel.OUTPUT:
            print(f"{Style.BRIGHT}{color}{message}")

    def log_internal(self, message: str, color: str = Fore.LIGHTBLACK_EX):
        if self.log_level >= AgentLogLevel.INTERNAL:
            print(f"{color}{Style.DIM}{message}")


agent_log = Logger()
