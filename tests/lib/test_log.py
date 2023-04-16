from mock import Mock

from surv_ai import AgentLogLevel, logger


def test_log_exception():
    logger._logger = Mock()
    logger.set_log_level(AgentLogLevel.EXCEPTION)
    logger.log_exception("test")

    logger._logger.exception.assert_called_once_with("test")


def test_log_error():
    logger._logger = Mock()
    logger.set_log_level(AgentLogLevel.ERROR)
    logger.log_error("test")

    logger._logger.error.assert_called_once_with("test")


def test_log_warning():
    logger._logger = Mock()
    logger.set_log_level(AgentLogLevel.WARNING)
    logger.log_warning("test")

    logger._logger.warning.assert_called_once_with("test")


def test_log_context():
    logger._logger = Mock()
    logger.set_log_level(AgentLogLevel.CONTEXT)
    logger.log_context("test")

    logger._logger.info.assert_called_once_with("\x1b[1m\x1b[37mtest\x1b[0m")


def test_log_output():
    logger._logger = Mock()
    logger.set_log_level(AgentLogLevel.OUTPUT)
    logger.log_output("test")

    logger._logger.info.assert_called_once_with("\x1b[1m\x1b[37mtest\x1b[0m")


def test_log_internal():
    logger._logger = Mock()
    logger.set_log_level(AgentLogLevel.INTERNAL)
    logger.log_internal("test")

    logger._logger.info.assert_called_once_with("\x1b[90m\x1b[2mtest\x1b[0m")
