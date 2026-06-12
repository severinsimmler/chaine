"""
chaine.optimization.utils
~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements utility functions for hyperparameter optimization.
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass

from chaine.typing import Labels, Sequence

# one half of a cross-validation split: data set and corresponding labels
Fold = tuple[list[Sequence], list[Labels]]


@dataclass(frozen=True)
class NumberSeries:
    start: int | float
    stop: int | float
    step: int | float

    def __repr__(self) -> str:
        return f"<NumberSeries (start={self.start}, stop={self.stop}, step={self.step})>"

    def __iter__(self) -> Iterator[int | float]:
        n = round((self.stop - self.start) / self.step)
        if n > 1:
            yield from (self.start + self.step * i for i in range(n + 1))
        elif n == 1:
            yield self.start


def cross_validation(
    dataset: list[Sequence],
    labels: list[Labels],
    k: int,
    seed: int | None = None,
) -> Iterator[tuple[Fold, Fold]]:
    """K-fold cross validation.

    Parameters
    ----------
    dataset : list[Sequence]
        Data set to split into k folds.
    labels : list[Labels]
        Labels to split into k folds.
    k : int
        Number of folds.
    seed : int | None, optional
        Random seed, by default None.

    Yields
    -------
    tuple[Fold, Fold]
        Train and test split.
    """
    # get indices of the examples
    indices = list(range(len(dataset)))

    # shuffle examples
    random.seed(seed)
    random.shuffle(indices)

    # split into k folds
    folds = [indices[i::k] for i in range(k)]

    # yield every fold split
    for i, fold in enumerate(folds):
        # get train and test indices
        test = set(fold)
        train = {index for j, other in enumerate(folds) if j != i for index in other}

        # yield train and test split
        yield (
            (
                [d for i, d in enumerate(dataset) if i in train],
                [l for i, l in enumerate(labels) if i in train],
            ),
            (
                [d for i, d in enumerate(dataset) if i in test],
                [l for i, l in enumerate(labels) if i in test],
            ),
        )


def downsample(
    dataset: list[Sequence],
    labels: list[Labels],
    n: int,
    seed: int | None = None,
) -> tuple[list[Sequence], list[Labels]]:
    """Downsample the given data set to the specified size.

    Parameters
    ----------
    dataset : list[Sequence]
        Data set to downsample.
    labels : list[Labels]
        Labels for the data set.
    n : int
        Number of samples to keep.
    seed : int | None, optional
        Random seed, by default None.

    Returns
    -------
    tuple[list[Sequence], list[Labels]]
        Downsampled data set and labels.

    Raises
    ------
    ValueError
        If number of instances in the data set is smaller than specified size.
    """
    if len(dataset) < n:
        raise ValueError("Data set is too small")

    # sample indices of the data set
    random.seed(seed)
    sample = set(random.sample(range(len(dataset)), n))

    # keep only instances of the sample
    return (
        [s for i, s in enumerate(dataset) if i in sample],
        [l for i, l in enumerate(labels) if i in sample],
    )
