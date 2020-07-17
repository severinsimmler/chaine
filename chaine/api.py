"""
chaine.api
~~~~~~~~~~

This module implements the high-level API
"""

from chaine.data import FeatureMatrix, Features, Sequence, Token
from chaine.typing import List, Optional, Iterable
from chaine.model import ConditionalRandomField


def matrix(tokens: Iterable[str], features: Optional[Features] = None) -> FeatureMatrix:
    """Create a feature matrix from plain tokens

    Parameters
    ----------
    tokens : Iterable[str]
        Iterable of plain token strings
    features : Optional[Features]
        Feature set to vectorize tokens

    Returns
    -------
    FeatureMatrix
        Matrix where rows represent tokens and columns features
    """
    # token objects
    tokens = [Token(index, token) for index, token in enumerate(tokens)]

    # token sequence
    sequence = Sequence(tokens)

    # new feature set
    if not features:
        features = Features()

    # return a feature matrix object
    return FeatureMatrix(sequence, features)


def process(dataset: Iterable[Iterable[str]], features: Optional[Features] = None):
    for instance in dataset:
        yield feature_matrix(instance, features)
