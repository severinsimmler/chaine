"""
chaine.data
~~~~~~~~~~~

This module provides basic data structures.
"""

from dataclasses import dataclass

import numpy as np

from chaine.typing import List, TokenGenerator


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
class Sentence:
    tokens: List[Token]

    def __getitem__(self, index: int) -> Token:
        return self.tokens[index]

    def __iter__(self) -> TokenGenerator:
        for token in self.tokens:
            yield token

    def __len__(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:
        return f"<Sentence: {self.tokens}>"


@dataclass
class FeatureMatrix:
    sentences: List[Sentence]

    def __post_init__(self):
        self._feature2index = {}

    def __iter__(self):
        for sentence in self.sentences:
            yield [self.vectorize(token) for token in sentence]

    def __repr__(self) -> str:
        return f"<FeatureMatrix: {len(self.sentences)} Sentences>"

    def vectorize(self, token: Token):
        features = []
        features.append(f"word.lower():{token.lower()}")
        features.append(f"word.is_upper:{token.is_upper}")
        features.append(f"word.is_title:{token.is_title}")
        features.append(f"word.is_digit:{token.is_digit}")
        self._add_features(features)
        return [self._feature2index[feature] for feature in features]

    def _add_features(self, features: List[str]):
        for feature in features:
            if feature not in self._feature2index:
                self._feature2index[feature] = len(self._feature2index)

    def numpy(self):
        matrix = list(self)
        return np.array(matrix)
