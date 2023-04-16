import logging
from enum import IntEnum

from colorama import Fore, Style


class AgentLogLevel(IntEnum):
    EXCEPTION = 0
    ERROR = 1
    WARNING = 2

    CONTEXT = 3
    OUTPUT = 4
    INTERNAL = 5


class Logger:
    def __init__(self, name="surv_ai", log_level=AgentLogLevel.ERROR):
        self.set_logger(name)
        self.set_log_level(log_level)

    def set_logger(self, name: str):
        self._logger = logging.getLogger(name)

    def set_log_level(self, log_level):
        self.log_level = log_level

        if log_level in [
            AgentLogLevel.CONTEXT,
            AgentLogLevel.OUTPUT,
            AgentLogLevel.INTERNAL,
        ]:
            self._logger.setLevel(logging.INFO)
        elif log_level in [
            AgentLogLevel.ERROR,
            AgentLogLevel.WARNING,
        ]:
            self._logger.setLevel(logging.WARNING)
        elif log_level == AgentLogLevel.EXCEPTION:
            self._logger.setLevel(logging.ERROR)

    def log_context(self, message: str, color: str = Fore.WHITE):
        if self.log_level >= AgentLogLevel.CONTEXT:
            self._logger.info(f"{Style.BRIGHT}{color}{message}{Style.RESET_ALL}")

    def log_output(self, message: str, color: str = Fore.WHITE):
        if self.log_level >= AgentLogLevel.OUTPUT:
            self._logger.info(f"{Style.BRIGHT}{color}{message}{Style.RESET_ALL}")

    def log_internal(self, message: str, color: str = Fore.LIGHTBLACK_EX):
        if self.log_level >= AgentLogLevel.INTERNAL:
            self._logger.info(f"{color}{Style.DIM}{message}{Style.RESET_ALL}")

    def log_exception(self, error):
        if self.log_level >= AgentLogLevel.EXCEPTION:
            self._logger.exception(error)

    def log_error(self, message: str):
        if self.log_level >= AgentLogLevel.ERROR:
            self._logger.error(f"{message}")

    def log_warning(self, message: str):
        if self.log_level >= AgentLogLevel.WARNING:
            self._logger.warning(f"{message}")


logger = Logger()
