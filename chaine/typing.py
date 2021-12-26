"""
chaine.typing
~~~~~~~~~~~~~

A collection of type hints.
"""

from os import PathLike
from pathlib import Path
from typing import Iterable, Union

Sequence = Iterable[dict[str, Union[str, int, float, bool]]]
Labels = Iterable[str]
Filepath = Union[Path, PathLike, str]
