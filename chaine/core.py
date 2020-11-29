"""
chaine.core
~~~~~~~~~~~

This module implements the high-level API
"""

from chaine.model import Trainer, CRF
from chaine.data import Token, Sequence
from chaine.typing import Iterable, Labels


def train(dataset: Iterable[Sequence], labels: Iterable[Labels], **kwargs) -> CRF:
    """Train a conditional random field

    Parameters
    ----------
    dataset : Iterable[Sequence]
        Dataset consisting of sequences of tokens
    labels : Iterable[Labels]
        Labels corresponding to the dataset


    Returns
    -------
    CRF
        A conditional random field fitted on the dataset
    """
    # generator expression to extract features
    features = (sequence.featurize() for sequence in dataset)

    # start training
    trainer = Trainer(**kwargs)
    trainer.train(features, labels, "model.crf")

    # load and return the trained model
    return CRF("model.crf")
