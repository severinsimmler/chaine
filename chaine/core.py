"""
chaine.core
~~~~~~~~~~~

This module implements the high-level API
"""

from chaine.crf import Trainer, Model
from chaine.data import LabelSequence, Token, TokenSequence
from chaine.typing import Labels, Dataset


def train(dataset: Dataset, labels: Labels, **kwargs) -> Model:
    """Train a conditional random field

    Parameters
    ----------
    dataset : Dataset
        Dataset consisting of sequences of features
    labels : Labels
        Labels corresponding to each instance in the dataset

    Returns
    -------
    CRF
        A conditional random field fitted on the dataset
    """
    # start training
    trainer = Trainer(**kwargs)
    trainer.train(dataset, labels, "model.crf")

    # load and return the trained model
    return Model("model.crf")


def featurize(dataset: Dataset):
    for sequence in dataset:
        yield sequence


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
