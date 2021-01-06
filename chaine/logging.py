"""
chaine.logging
~~~~~~~~~~~~~~

This module implements a basic logger
"""

import logging
import sys


class Logger(logging.Logger):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

    def __init__(self, name: str):
        """Basic logger

        Parameters
        ----------
        name : str
            Name of the logger
        """
        super().__init__(name)

        # stream handler to stdout
        self._stream_handler = logging.StreamHandler(sys.stdout)
        self._stream_handler.setFormatter(self.formatter)
        self.addHandler(self._stream_handler)

        self.setLevel(self.INFO)

    def debug(self, message: str):
        """Debug log message

        Parameters
        ----------
        message : str
            Message to log
        """
        if self.isEnabledFor(self.DEBUG):
            self._log(self.DEBUG, message, ())

    def info(self, message: str):
        """Info log message

        Parameters
        ----------
        message : str
            Message to log
        """
        if self.isEnabledFor(self.INFO):
            self._log(self.INFO, message, ())

    def warning(self, message: str):
        """Warning log message

        Parameters
        ----------
        message : str
            Message to log
        """
        if self.isEnabledFor(self.WARNING):
            self._log(self.WARNING, message, ())

    def error(self, message: str):
        """Error log message

        Parameters
        ----------
        message : str
            Message to log
        """
        if self.isEnabledFor(self.ERROR):
            self._log(self.ERROR, message, ())

    @property
    def log_level(self) -> int:
        """Log level

        Returns
        -------
        int
            The current log level.
        """
        return self._log_level

    @log_level.setter
    def log_level(self, level: int):
        """Set log level

        Paramters
        ---------
        level : int
            Log level
        """
        self._log_level = level
        self.setLevel(self._log_level)

    @property
    def formatter(self):
        """Logging format

        Example
        -------
        [1970-01-01 00:00:00,000] [INFO] Hello world!
        """
        return logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    def __repr__(self):
        return f"<Logger: {self.name}>"
