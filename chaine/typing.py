"""
chaine.typing
~~~~~~~~~~~~~

A collection of type hints.
"""

from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Iterator

Sequence = Iterable[dict[str, str | int | float | bool]]
Labels = Iterable[str]
Filepath = Path | PathLike | str
Sentence = list[str]
Tags = list[str]
Features = dict[str, float | int | str | bool]
Dataset = dict[str, dict[str, Any]]
