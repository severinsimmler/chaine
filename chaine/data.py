"""
chaine.data
~~~~~~~~~~~

This module provides basic data structures
"""

from abc import abstractmethod, ABC
from dataclasses import dataclass

from chaine.typing import FeatureGenerator, List, Iterable, Any, Generator


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


@dataclass
class _Sequence(ABC):
    items: Iterable[Any]

    def __getitem__(self, index: int):
        """Get item of the sequence by index"""
        return self.items[index]

    def __iter__(self) -> Generator[Any, None, None]:
        """Iterate over the items of the sequence"""
        for item in self.items:
            yield item

    def __len__(self) -> int:
        """Number of items in the sequence"""
        return len(self.items)

    @abstractmethod
    def __repr__(self) -> str:
        """Representation of the sequence"""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """String representation of the sequence"""
        pass


class TokenSequence(_Sequence):
    def __repr__(self) -> str:
        """Representation of the sequence"""
        return f"<TokenSequence: {self.items}>"

    def __str__(self) -> str:
        """String representation of the sequence

        Note
        ----
        Overwrite this method for custom detokenization
        """
        return " ".join(token.text for token in self.items)

    @property
    def indices(self) -> List[int]:
        """Indices of the token sequence"""
        return [token.index for token in self.items]

    def featurize(self) -> FeatureGenerator:
        """Extract features from tokens of the sequence

        Note
        ----
        Overwrite this method for custom feature extraction – which is generally
        recommended as the default features may result in very low accuracy.

        One token is represented as a set of strings, each string is a unique feature,
        e.g. the string representation of the current token.
        """
        for token in self.items:
            features = {f"token.lower={token.lower()}",
                f"token.is_upper={token.is_upper}",
                f"token.is_title={token.is_title}",
                f"token.is_digit={token.is_digit}",
            }

            if token.index > 0:
                left_token = self.items[token.index - 1]
                features.update(
                    {
                        f"-1:token.lower={left_token.lower()}",
                        f"-1:token.is_title={left_token.is_title}",
                        f"-1:token.is_upper={left_token.is_upper}",
                    }
                )
            else:
                features.add("BOS=True")

            if token.index < max(self.indices):
                right_token = self.items[token.index + 1]
                features.update(
                    {
                        f"+1:token.lower={right_token.lower()}",
                        f"+1:token.is_title={right_token.is_title}",
                        f"+1:token.is_upper={right_token.is_upper}",
                    }
                )
            else:
                features.add("EOS=True")
            yield features


@dataclass
class LabelSequence(_Sequence):
    def __post_init__(self):
        # each label must be a string
        self.items = [str(item) for item in self.items]

    def __repr__(self):
        """Representation of a label sequence"""
        return f"<LabelSequence: {self.items}>"

    def __str__(self):
        """String representation of a label sequence"""
        return ", ".join(self.items)

    def __eq__(self, other: "LabelSequence") -> bool:
        """True if two label sequences are equal"""
        return all(a == b for a, b in zip(self, other))

    @property
    def distinct(self):
        """Distinct labels in the sequence"""
        return set(self.items)
