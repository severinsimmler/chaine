"""
chaine.typing
~~~~~~~~~~~~~

A collection of type hints.
"""

from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union

Sequence = Iterable[dict[str, Union[str, int, float, bool]]]
Labels = Iterable[str]
Filepath = Union[Path, PathLike, str]
Sentence = list[str]
Tags = list[str]
Features = dict[str, Union[float, int, str, bool]]
Dataset = dict[str, dict[str, Any]]
