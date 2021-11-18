"""
chaine.typing
~~~~~~~~~~~~~

A collection of type hints.
"""

from pathlib import Path
from os import PathLike
from typing import Any, Generator, Iterable, Optional, Union

Sequence = Iterable[dict[str, Union[str, int, float, bool]]]
Labels = Iterable[str]
Filepath = Union[Path, PathLike, str]
