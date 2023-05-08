import logging

from enum import IntEnum

from colorama import Fore, Style


_logger = logging.getLogger(__name__)


class AgentLogLevel(IntEnum):
    EXCEPTION = 0
    ERROR = 1
    WARNING = 2

    CONTEXT = 3
    OUTPUT = 4
    INTERNAL = 5


class Logger:
    def __init__(self, log_level=AgentLogLevel.OUTPUT):
        self.log_level = log_level

    def set_log_level(self, log_level):
        self.log_level = log_level

        if log_level in [
            AgentLogLevel.CONTEXT,
            AgentLogLevel.OUTPUT,
            AgentLogLevel.INTERNAL,
        ]:
            logging.basicConfig(level=logging.INFO)

    def log_context(self, message: str, color: str = Fore.WHITE):
        if self.log_level >= AgentLogLevel.CONTEXT:
            _logger.info(f"{Style.BRIGHT}{color}{message}{Style.RESET_ALL}")

    def log_output(self, message: str, color: str = Fore.WHITE):
        if self.log_level >= AgentLogLevel.OUTPUT:
            _logger.info(f"{Style.BRIGHT}{color}{message}{Style.RESET_ALL}")

    def log_internal(self, message: str, color: str = Fore.LIGHTBLACK_EX):
        if self.log_level >= AgentLogLevel.INTERNAL:
            _logger.info(f"{color}{Style.DIM}{message}{Style.RESET_ALL}")

    def log_exception(self, error):
        _logger.exception(error)


logger = Logger()
