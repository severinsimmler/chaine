"""
chaine.data
~~~~~~~~~~~

This module provides basic data structures
"""

from dataclasses import dataclass

from chaine.typing import FeatureGenerator, List, TokenGenerator


@dataclass
class Token:
    index: int
    text: str

    def __len__(self) -> int:
        return len(self.text)

    def __repr__(self) -> str:
        return f"<Token {self.index}: {self.text}>"

    def __str__(self) -> str:
        return self.text

    def lower(self) -> str:
        return self.text.lower()

    @property
    def is_digit(self) -> bool:
        return self.text.isdigit()

    @property
    def is_lower(self) -> bool:
        return self.text.islower()

    @property
    def is_title(self) -> bool:
        return self.text.istitle()

    @property
    def is_upper(self) -> bool:
        return self.text.isupper()


@dataclass
class Sequence:
    tokens: List[Token]

    def __getitem__(self, index: int) -> Token:
        return self.tokens[index]

    def __iter__(self) -> TokenGenerator:
        for token in self.tokens:
            yield token

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        return f"<Sequence: {self.tokens}>"


class Features:
    _feature2index = {}

    def index(self, name: str) -> int:
        if name not in self._feature2index:
            self._feature2index[name] = len(self._feature2index)
        return self._feature2index[name]

    def vectorize(self, token: Token) -> List[int]:
        return [
            self.index(feature)
            for feature in [
                f"word.lower():{token.lower()}",
                f"word.is_upper:{token.is_upper}",
                f"word.is_title:{token.is_title}",
                f"word.is_digit:{token.is_digit}",
            ]
        ]


@dataclass
class FeatureMatrix:
    sequence: Sequence
    features: Features

    def __iter__(self) -> FeatureGenerator:
        for token in self.sequence:
            yield self.features.vectorize(token)

    def __repr__(self) -> str:
        return f"<FeatureMatrix: {len(self.sequence)} Tokens>"


class Labels:
    pass


class Parameters:
    pass