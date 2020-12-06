"""
chaine.data
~~~~~~~~~~~

This module provides basic data structures
"""

import re
from dataclasses import dataclass

from chaine.typing import Iterable


@dataclass
class Token:
    index: int
    text: str

    def __len__(self) -> int:
        """Number of characters"""
        return len(self.text)

    def __repr__(self) -> str:
        """Representation of the token"""
        return f"<Token {self.index}: {self.text}>"

    def __str__(self) -> str:
        """String representation of the token"""
        return self.text

    def lower(self) -> str:
        """Lower case of the token"""
        return self.text.lower()

    @property
    def shape(self) -> str:
        text = re.sub("[A-Z]", "X", self.text)
        text = re.sub("[a-z]", "x", text)
        return re.sub("[0-9]", "d", text)

    @property
    def is_digit(self) -> bool:
        """True if token is a digit, False otherwise"""
        return self.text.isdigit()

    @property
    def is_lower(self) -> bool:
        """True if token is lower case, False otherwise"""
        return self.text.islower()

    @property
    def is_title(self) -> bool:
        """True if first letter is upper case, False otherwise"""
        return self.text.istitle()

    @property
    def is_upper(self) -> bool:
        """True if token is upper case, False otherwise"""
        return self.text.isupper()


class TokenSequence:
    def __init__(self, tokens):
        if not all(isinstance(token, Token) for token in tokens):
            tokens = [Token(index, text) for index, text in enumerate(tokens)]
        self.tokens = tokens

    def __iter__(self):
        for token in self.tokens:
            yield token
