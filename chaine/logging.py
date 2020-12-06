"""
chaine.logging
~~~~~~~~~~~~~~

This module implements a basic logger and a parser for CRFsuite
"""

import logging
import re
import sys

from chaine.typing import Optional


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
        """Log level.

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


class LogMessage:
    """CRFsuite log message

    Attributes
    ----------
    iteration : Optional[str]
        Current number of iterations
    loss : Optional[str]
        Current loss score
    """

    def __init__(self):
        self.iteration = None
        self.loss = None

    def __str__(self) -> str:
        return f"Iteration: {self.iteration}\tLoss: {self.loss}"


class LogParser:
    """Parser for CRFsuite's logfile

    Attributes
    ----------
    message : LogMessage
        Log message with current iteration and loss
    """

    def __init__(self):
        self.message = LogMessage()

    def parse(self, line: str) -> Optional[str]:
        """Parse one line of the logs

        Parameters
        ----------
        line : str
            One line of CRFsuite's logs

        Returns
        -------
        str
            Formatted log message with latest iteration and loss
        """
        if (m := re.match(r"\*{5} (?:Iteration|Epoch) #(\d+) \*{5}\n", line)) :
            self.message.iteration = m.group(1)
        elif (m := re.match(r"Loss: (\d+\.\d+)", line)) :
            self.message.loss = m.group(1)
            if self.message.iteration:
                text = str(self.message)
                self.message = LogMessage()
                return text
