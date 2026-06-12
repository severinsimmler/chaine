"""
chaine.logging
~~~~~~~~~~~~~~

This module implements a basic logger.
"""

import logging
import sys
from logging import DEBUG, ERROR, INFO, WARNING, Formatter, StreamHandler

LEVELS = {"DEBUG": DEBUG, "INFO": INFO, "WARNING": WARNING, "ERROR": ERROR}

DEFAULT_FORMAT = Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
DEBUG_FORMAT = Formatter("[%(asctime)s] %(name)s:%(lineno)d [%(levelname)s] %(message)s")


def _resolve_level(level: str | int) -> int:
    """Translate a string level (e.g. 'INFO') to its integer value."""
    if isinstance(level, str):
        return LEVELS[level.upper()]
    return level


def _set_level(logger: logging.Logger, level: int):
    """Set level on the logger and all its handlers, adjusting the format."""
    logger.setLevel(level)

    for handler in logger.handlers:
        handler.setLevel(level)

        # log more details when in debug mode
        handler.setFormatter(DEBUG_FORMAT if level < INFO else DEFAULT_FORMAT)


class Logger:
    def __init__(self, name: str):
        """Basic logger

        Parameters
        ----------
        name : str
            Name of the logger
        """
        self.name = name

        # return a logger with the specified name, creating it if necessary
        self._logger = logging.getLogger(name)

        # stream handler to stdout (only once, even if the logger already exists)
        if not self._logger.handlers:
            self._logger.addHandler(StreamHandler(sys.stdout))

        # set level of both the logger and the handler to INFO by default
        self.set_level("INFO")

    def set_level(self, level: str | int):
        _set_level(self._logger, _resolve_level(level))

    def debug(self, message: str):
        """Debug log message

        Parameters
        ----------
        message : str
            Message to log
        """
        self._logger.debug(message)

    def info(self, message: str):
        """Info log message

        Parameters
        ----------
        message : str
            Message to log
        """
        self._logger.info(message)

    def warning(self, message: str):
        """Warning log message

        Parameters
        ----------
        message : str
            Message to log
        """
        self._logger.warning(message)

    def error(self, message: str | Exception):
        """Error log message

        Parameters
        ----------
        message : str | Exception
            Message to log (logs the stacktrace if it is an exception)
        """
        self._logger.error(message, exc_info=isinstance(message, Exception))

    @property
    def in_debug_mode(self) -> bool:
        """Checks if the logger's level is DEBUG

        Returns
        -------
        bool
            True, if logger is in DEBUG mode, False otherwise
        """
        return self._logger.level == DEBUG

    @property
    def level(self) -> int:
        """Returns the current log level

        Returns
        -------
        int
            Log level.
        """
        return self._logger.level

    def __repr__(self):
        return f"<Logger: {self.name} ({self.level})>"


def get_logger(name: str) -> logging.Logger:
    """Gets the specified logger object

    Parameters
    ----------
    name : str
        Name of the module to get the logger for

    Returns
    -------
    logging.Logger
        Logger of the specified module
    """
    return logging.getLogger(name)


def logger_exists(name: str) -> bool:
    """Checks if a logger exists for the specified module

    Parameters
    ----------
    name : str
        Name of the module to check the logger for

    Returns
    -------
    bool
        True if logger exists, False otherwise
    """
    return logging.getLogger(name).hasHandlers()


def set_level(name: str, level: int | str):
    """Sets log level for the specified logger

    Parameters
    ----------
    name : str
        Name of the module
    level : int | str
        Level to set
    """
    _set_level(logging.getLogger(name), _resolve_level(level))


def set_verbosity(level: int):
    """Sets verbosity to the given level

    Parameters
    ----------
    level : int
        Log only errors (0), info (1) or even debug messages (2)
    """
    levels = {0: "ERROR", 1: "INFO", 2: "DEBUG"}
    if level in levels:
        set_level("chaine._core.crf", levels[level])
        set_level("chaine.crf", levels[level])
