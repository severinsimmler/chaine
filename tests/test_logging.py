from chaine import logging
from chaine.logging import DEBUG, WARNING


def test_logger():
    logger = logging.Logger("test")

    assert logger.name == "test"

    logger.level = DEBUG
    assert logger.level == DEBUG
    logger.set_level(WARNING)
    assert logger == WARNING
