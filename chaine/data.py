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

    def __str__(self) -> str:
        return " ".join(self.tokens)

    @property
    def indices(self) -> List[int]:
        return [token.index for token in self.tokens]

    def featurize(self) -> FeatureGenerator:
        for token in self.tokens:
            features = {
                "bias=1.0",
                f"token.lower()={token.lower()}",
                f"token.is_upper()={token.is_upper}",
                f"token.is_title()={token.is_title}",
                f"token.is_digit()={token.is_digit}",
            }

            if token.index > 0:
                left_token = self.tokens[token.index - 1]
                features.update(
                    {
                        f"-1:token.lower()={left_token.lower()}",
                        f"-1:token.is_title()={left_token.is_title}",
                        f"-1:token.is_upper()={left_token.is_upper}",
                    }
                )
            else:
                features.add("BOS=True")

            if token.index < max(self.indices):
                right_token = self.tokens[token.index + 1]
                features.update(
                    {
                        f"+1:token.lower()={right_token.lower()}",
                        f"+1:token.is_title()={right_token.is_title}",
                        f"+1:token.is_upper()={right_token.is_upper}",
                    }
                )
            else:
                features.add("EOS=True")
            yield features