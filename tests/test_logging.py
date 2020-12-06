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


def test_log_message():
    message = logging.LogMessage()

    assert message.iteration is None
    assert message.loss is None

    message.iteration = "1"
    message.loss = "1000.0"
    assert message.iteration == "1"
    assert message.loss == "1000.0"
    assert str(message) == "Iteration: 1\tLoss: 1000.0"


def test_log_parser():
    parser = logging.LogParser()

    assert isinstance(parser.message, logging.LogMessage)

    text = parser.parse("Irrelevant message")
    assert text is None
    assert parser.message.iteration is None
    assert parser.message.loss is None

    text = parser.parse("***** Iteration #1 *****\n")
    assert text is None
    assert parser.message.iteration == "1"
    assert parser.message.loss is None

    text = parser.parse("Loss: 1000.0")
    assert text == "Iteration: 1\tLoss: 1000.0"
    assert parser.message.iteration is None
    assert parser.message.loss is None
