"""
chaine.typing
~~~~~~~~~~~~~

A collection of type hints
"""

from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Union

Labels = Iterable[Iterable[str]]
Dataset = Iterable[Iterable[str]]
Filepath = Union[Path, str]
Sequence = List[Union[Set[str], Dict[str, Union[int, float, str, bool]]]]
