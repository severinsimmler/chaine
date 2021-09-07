"""
chaine.typing
~~~~~~~~~~~~~

A collection of type hints.
"""

from pathlib import Path
from os import PathLike
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Union

Sequence = Iterable[Dict[str, Union[str, int, float, bool]]]
Labels = Iterable[str]
Filepath = Union[Path, PathLike, str]
