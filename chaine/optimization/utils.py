"""
chaine.optimization.utils
~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements utility functions for hyperparameter optimization.
"""

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from chaine.typing import Labels, Sequence


@dataclass
class NumberSeries(Iterable):
    start: int
    stop: int
    step: int | float

    def __repr__(self) -> str:
        return f"<NumberSeries (start={self.start}, stop={self.stop}, step={self.step})>"

    def __iter__(self) -> Iterator[int | float]:
        n = int(round((self.stop - self.start) / float(self.step)))
        if n > 1:
            yield from [self.start + self.step * i for i in range(n + 1)]
        elif n == 1:
            yield self.start


def cross_validation(
    dataset: Iterable[Sequence], labels: Iterable[Labels], k: int, seed: int | None = None
) -> Iterator[tuple[tuple[Iterable[Sequence], Iterable[Labels]]]]:
    """K-fold cross validation.

    Parameters
    ----------
    dataset : Iterable[Sequence]
        Data set to split into k folds.
    labels : Iterable[Labels]
        Labels to split into k folds.
    k : int
        Number of folds.
    shuffle : bool, optional
        True if data set should be shuffled first, by default True.

    Yields
    -------
    Iterator[tuple[tuple[Iterable[Sequence], Iterable[Labels]]]]
        Train and test set.
    """
    # get indices of the examples
    indices = list(range(len(dataset)))

    # shuffle examples
    random.seed(seed)
    random.shuffle(indices)

    # split into k folds
    folds = [indices[i::k] for i in range(k)]

    # yield every fold split
    for i in range(k):
        # get train and test split
        test = folds[i]
        train = [s for x in [fold for fold in folds if fold != test] for s in x]

        # yield train and test split
        yield (
            [d for i, d in enumerate(dataset) if i in train],
            [l for i, l in enumerate(labels) if i in train],
        ), (
            [d for i, d in enumerate(dataset) if i in test],
            [l for i, l in enumerate(labels) if i in test],
        )


def downsample(
    dataset: Iterable[Sequence], labels: Iterable[Labels], n: int, seed: int | None = None
) -> tuple[Iterable[Sequence], Iterable[Labels]]:
    """Downsample the given data set to the specified size.

    Parameters
    ----------
    dataset : Iterable[Sequence]
        Data set to downsample.
    labels : Iterable[Labels]
        Labels for the data set.
    n : int
        Number of samples to keep.
    seed : int | None, optional
        Random seed, by default None.

    Returns
    -------
    tuple[Iterable[Sequence], Iterable[Labels]]
        Downsampled data set and labels.

    Raises
    ------
    ValueError
        If number of instances in the data set is smaller than specified size.
    """
    if len(dataset) < n:
        raise ValueError("Data set is too small")

    # get indices of the data set
    indices = list(range(len(dataset)))

    # sample indices
    random.seed(seed)
    sample = set(random.sample(indices, n))

    # keep only instances of the sample
    dataset = [s for i, s in enumerate(dataset) if i in sample]
    labels = [l for i, l in enumerate(labels) if i in sample]

    return dataset, labels
