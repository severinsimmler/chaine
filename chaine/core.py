"""
chaine.core
~~~~~~~~~~~

This module implements the high-level API
"""

from chaine.crf import Model, Trainer
from chaine.typing import Dataset, Labels


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
