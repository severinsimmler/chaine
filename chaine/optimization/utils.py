import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Optional, Union

from chaine.typing import Labels, Sequence


def cross_validation(
    dataset: Iterable[Sequence], labels: Iterable[Labels], n: int, shuffle: bool = True
) -> Iterator[tuple[tuple[Iterable[Sequence], Iterable[Labels]]]]:
    """K-fold cross validation.

    Parameters
    ----------
    dataset : Iterable[Sequence]
        Data set to split into k folds.
    labels : Iterable[Labels]
        Labels to split into k folds.
    n : int
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

    if shuffle:
        # optionally shuffle examples
        random.shuffle(indices)

    # split into k folds
    folds = [set(indices[i::n]) for i in range(n)]

    # yield every fold split
    for i in range(n):
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


@dataclass
class NumberSeries(Iterable):
    start: int
    stop: int
    step: Union[int, float]

    def __repr__(self) -> str:
        return f"<NumberSeries (start={self.start}, stop={self.stop}, step={self.step})>"

    def __iter__(self) -> Iterator[Union[int, float]]:
        n = int(round((self.stop - self.start) / float(self.step)))
        if n > 1:
            yield from [self.start + self.step * i for i in range(n + 1)]
        elif n == 1:
            yield self.start


def downsample(
    dataset: Iterable[Sequence], labels: Iterable[Labels], n: int, seed: Optional[int] = None
) -> tuple[Iterable[Sequence], Iterable[Labels]]:
    """[summary]

    Parameters
    ----------
    dataset : Iterable[Sequence]
        [description]
    labels : Iterable[Labels]
        [description]
    n : int
        [description]
    seed : Optional[int], optional
        [description], by default None

    Returns
    -------
    tuple[Iterable[Sequence], Iterable[Labels]]
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    if len(dataset) < n:
        raise ValueError("Data set is too small")

    # tbd
    indices = list(range(len(dataset)))

    # tbd
    random.seed(seed)
    sample = set(random.sample(indices, n))

    # tbd
    dataset = [s for i, s in enumerate(dataset) if i in sample]
    labels = [l for i, l in enumerate(labels) if i in sample]

    return dataset, labels
