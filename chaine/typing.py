"""
chaine.typing
~~~~~~~~~~~~~

A collection of type hints.
"""

from collections.abc import Iterable
from os import PathLike
from typing import Any

Sequence = Iterable[dict[str, str | int | float | bool]]
Labels = Iterable[str]
Filepath = str | PathLike
Sentence = list[str]
Tags = list[str]
Features = dict[str, float | int | str | bool]
Dataset = dict[str, dict[str, Any]]
