"""
chaine.utils
~~~~~~~~~~~~

This module implements general helper functions
"""

import re


class _LogMessage:
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


class _LogParser:
    """Parser for CRFsuite's logfile

    Attributes
    ----------
    message : _LogMessage
        Log message with current iteration and loss
    """

    def __init__(self):
        self.message = _LogMessage()

    def parse(self, line: str) -> str:
        """Parse line of the logfile

        Parameters
        ----------
        line : str
            One line of CRFsuite's logfile

        Returns
        -------
        str
            Formatted log message with current iteration and loss
        """
        if (m := re.match(r"\*{5} (?:Iteration|Epoch) #(\d+) \*{5}\n", line)):
            self.message.iteration = m.group(1)
        elif (m := re.match(r"Loss: (\d+\.\d+)", line)):
            self.message.loss = m.group(1)
            text = str(self.message)
            self.message = _LogMessage()
            return text
