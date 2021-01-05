"""
chaine.data
~~~~~~~~~~~

This module provides basic data structures
"""

import re
from dataclasses import dataclass


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

    @property
    def features(self):
        return {"num_characters": len(self),
        "text": self.lower(),
        "shape": self.shape,
        "is_digit": self.is_digit,
        "is_lower": self.is_lower,
        "is_title": self.is_title,
        "is_upper": self.is_upper}
