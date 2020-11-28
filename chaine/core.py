"""
chaine.core
~~~~~~~~~~~

This module implements the high-level API
"""

from chaine.model import Trainer, CRF
from chaine.data import Token, Sequence
from chaine.typing import FeatureGenerator, List


def featurize(tokens: List[str]) -> FeatureGenerator:
    """Featurize a sequence of tokens

    Parameters
    ----------
    tokens : List[str]
        Sequence of tokens to generate features for

    Returns
    -------
    FeatureGenerator
        One feature set at a time
    """
    tokens = [Token(index, token) for index, token in enumerate(tokens)]
    for features in Sequence(tokens).featurize():
        yield features


def train(dataset: List[List[str]], labels: List[List[str]], **kwargs) -> CRF:
    """Train a linear-chain conditional random field

    Parameters
    ----------
    dataset : List[List[str]]
        Dataset consisting of sequences of tokens
    labels : List[List[str]]
        Labels corresponding to the dataset
    
    Returns
    -------
    CRF
        A conditional random field fitted on the dataset
    """
    features = [featurize(sequence) for sequence in dataset]

    trainer = Trainer("lbfgs", **kwargs)
    trainer.train(features, labels, "model.crf")

    return CRF("model.crf")
