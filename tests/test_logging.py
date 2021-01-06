from logging import Formatter

from chaine import logging


def test_logger():
    logger = logging.Logger("test")

    assert logger.name == "test"
    assert isinstance(logger.formatter, Formatter)

    logger.log_level = logger.DEBUG
    assert logger.log_level == logger.DEBUG
    logger.log_level = logger.WARNING
    assert logger.log_level == logger.WARNING
