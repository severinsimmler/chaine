"""
chaine.core
~~~~~~~~~~~

This module implements the high-level API
"""

from chaine.model import Trainer, CRF
from chaine.data import LabelSequence, Token, TokenSequence
from chaine.typing import Iterable, Labels


def train(dataset: Iterable[TokenSequence], labels: Iterable[Labels], **kwargs) -> CRF:
    """Train a conditional random field

    Parameters
    ----------
    dataset : Iterable[TokenSequence]
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


def token_sequences(dataset: Iterable[Iterable[str]]):
    # TODO: merge this and the other function in a dataset function?
    # TODO: unit test
    # TODO: remove token initialization when bug is fixed in sequence class
    # TODO: docstring: Transforms may be the wrong word because the original
    #                  dataset will not be changed
    """Transforms an iterable dataset to a generator with TokenSequences

    Parameters
    ----------
    dataset : Iterable[Iterable[str]]
        Dataset consisting of iterable dataset entries with strings

    Returns
    -------
        Generator
            Generator of TokenSequences for every dataset entry
    """
    return (
        TokenSequence([Token(index, text) for index, text in enumerate(tokens)])
        for tokens in dataset
    )


def label_sequences(dataset: Iterable[Iterable[str]]):
    # TODO: unit test
    # TODO: docstring: Transforms may be the wrong word because the original
    #                  dataset will not be changed
    """Transforms an iterable dataset to a generator with LabelSequences

    Parameters
    ----------
    dataset : Iterable[Iterable[str]]
        Dataset consisting of iterable dataset entries with strings

    Returns
    -------
        Generator
            Generator of LabelSequences for every dataset entry
    """
    return (LabelSequence(labels) for labels in dataset)
