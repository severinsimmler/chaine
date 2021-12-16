from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Union

from chaine.typing import Labels, Sequence


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


def cross_validation(dataset, labels, n) -> tuple[Iterable[Sequence], Iterable[Labels]]:
    pass
