"""
chaine.logging
~~~~~~~~~~~~~~

This module implements a basic logger
"""

import logging
import sys
from logging import Formatter, StreamHandler
from typing import Union

DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40
LEVELS = {"DEBUG": DEBUG, "INFO": INFO, "WARNING": WARNING, "ERROR": ERROR}

DEFAULT_FORMAT = Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
DEBUG_FORMAT = Formatter("[%(asctime)s] %(name)s:%(lineno)d [%(levelname)s] %(message)s")


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

        # stream handler to stdout
        self._stream_handler = StreamHandler(sys.stdout)
        self._logger.addHandler(self._stream_handler)

        # set level of both the logger and the handler to INFO by default
        self.set_level("INFO")

    def set_level(self, level: Union[str, int]):
        # translate string to integer
        if isinstance(level, str):
            level = LEVELS[level.upper()]

        # set the logger's level
        self._logger.setLevel(level)

        # and all handlers
        for handler in self._logger.handlers:
            handler.setLevel(level)

            # optionally change the formatter (log more when in debug mode)
            if level < INFO:
                handler.setFormatter(DEBUG_FORMAT)
            else:
                handler.setFormatter(DEFAULT_FORMAT)

    def debug(self, message: str):
        """Debug log message

        Parameters
        ----------
        message : str
            Message to log
        """
        if self._logger.isEnabledFor(DEBUG):
            self._logger._log(DEBUG, message, ())

    def info(self, message: str):
        """Info log message

        Parameters
        ----------
        message : str
            Message to log
        """
        if self._logger.isEnabledFor(INFO):
            self._logger._log(INFO, message, ())

    def warning(self, message: str):
        """Warning log message

        Parameters
        ----------
        message : str
            Message to log
        """
        if self._logger.isEnabledFor(WARNING):
            self._logger._log(WARNING, message, ())

    def error(self, message: Union[str, Exception]):
        """Error log message

        Parameters
        ----------
        message : str
            Message to log
        """
        if self._logger.isEnabledFor(ERROR):
            if isinstance(message, Exception):
                # log stacktrace if message is an exception
                self._logger._log(ERROR, message, (), exc_info=True)
            else:
                self._logger._log(ERROR, message, ())

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


def set_level(name: str, level: Union[int, str]):
    """Sets log level for the specified logger

    Parameters
    ----------
    name : str
        Name of the module

    level : Union[int, str]
        Level to set
    """
    logger = logging.getLogger(name)

    # translate string to integer
    if isinstance(level, str):
        level = LEVELS[level.upper()]

    # set the logger's level
    logger.setLevel(level)

    # and all handlers
    for handler in logger.handlers:
        handler.setLevel(level)

        # optionally change the formatter (log more when in debug mode)
        if level < INFO:
            handler.setFormatter(DEBUG_FORMAT)
        else:
            handler.setFormatter(DEFAULT_FORMAT)
