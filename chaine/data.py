import json
from pathlib import Path

from chaine.typing import (
    Callable,
    FeatureGenerator,
    Filepath,
    List,
    Matrix,
    MatrixGenerator,
    Optional,
    SentenceGenerator,
    TokenGenerator,
    Vector,
)


class Token:
    def __init__(self, index: int, text: str, label: Optional[Label] = None):
        self.index = index
        self.text = text
        self.label = label

    def __repr__(self) -> str:
        return f"<Token {self.index}: {self.text}>"

    def __len__(self) -> int:
        return len(self.text)

    def lower(self) -> str:
        return self.text.lower()

    def isupper(self) -> bool:
        return self.text.isupper()

    def istitle(self) -> bool:
        return self.text.istitle()

    def isdigit(self) -> bool:
        return self.text.isdigit()

    @property
    def vector(self) -> Vector:
        if not hasattr(self, "_vector"):
            raise ValueError
        else:
            return self._vector

    @vector.setter
    def vector(self, vector: Vector):
        self._vector = vector


class Sentence:
    def __init__(self, tokens: List[Token], features: "Features"):
        for i, vector in enumerate(features.vectorize(tokens)):
            tokens[i].vector = vector
        self.tokens = tokens

    def __repr__(self) -> str:
        return f"<Sentence: {' '.join([token.text for token in self.tokens])}>"

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> Token:
        return self.tokens[index]

    def __iter__(self) -> TokenGenerator:
        for token in self.tokens:
            yield token

    @property
    def matrix(self) -> Matrix:
        return [token.vector for token in self.tokens]


class Features:
    def __init__(self):
        self._values = dict()

    def __repr__(self) -> str:
        return f"<Features: {len(self._values)}>"

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, feature: str) -> int:
        try:
            return self._values[feature]
        except KeyError:
            raise KeyError(f"Feature '{feature}' is not available.")

    def vectorize(self, sentence: Sentence) -> FeatureGenerator:
        for token in sentence:
            features = list()
            features.append(f"word.lower():{token.lower()}")
            features.append(f"word.isupper():{token.isupper()}")
            features.append(f"word.istitle():{token.istitle()}")
            features.append(f"word.isdigit():{token.isdigit()}")
            self._add_features(features)
            yield [self._values[feature] for feature in features]

    def _add_features(self, features):
        for feature in features:
            if feature not in self._values:
                self._values[feature] = len(self._values)

    @classmethod
    def load(self, filepath: Filepath) -> "Features":
        with Path(filepath).open("r", encoding="utf-8") as features:
            values = json.load(features)
            features = cls()
            features._values = values
            return features

    def save(self, filepath: Filepath):
        with Path(filepath).open("w", encoding="utf-8") as features:
            json.dump(self._values, features, indent=4, ensure_ascii=False)


class Dataset:
    def __init__(self, sentences: List[Sentence], features: Features):
        self.sentences = sentences
        self.features = features

    def __repr__(self) -> str:
        return f"<Dataset: {len(self.sentences)} sentences>"

    def __len__(self) -> int:
        return len(self.sentences)

    def __iter__(self) -> MatrixGenerator:
        for sentence in self.sentences:
            return sentence.matrix

    def __getitem__(self, index: int) -> Sentence:
        return self.sentences[index]

    @classmethod
    def load(cls, corpus: Filepath, features: Optional[Filepath] = None) -> "Dataset":
        if features:
            features = Features.load(features)
        else:
            features = Features()
        sentences = list(cls._read_dataset(corpus, features))
        return cls(sentences, features)

    @staticmethod
    def _read_dataset(filepath: Filepath, features: Features) -> SentenceGenerator:
        with Path(filepath).open("r", encoding="utf-8") as dataset:
            sentence = list()
            index = 0
            for row in dataset:
                if row:
                    token, label = row.strip().split(" ")
                    token = Token(index, token)
                    sentence.append(token)
                    index += 1
                else:
                    yield Sentence(sentence, features)
                    sentence = list()
                    index = 0
            if sentence:
                yield Sentence(sentence, features)


class Label:
    def __init__(self, gold: str):
        self.gold = gold
        self.values = ["*"] + values

    def __getitem__(self, index: int):
        return self.values[index]
