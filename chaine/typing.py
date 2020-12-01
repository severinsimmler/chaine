"""
chaine.typing
~~~~~~~~~~~~~

A collection of type hints
"""

from pathlib import Path as _Path
from typing import List, Optional, Generator, Iterable, Union, Any, Set

FeatureGenerator = Generator[List[str], None, None]
TokenGenerator = Generator["Token", None, None]
Labels = Iterable[str]
Dataset = Iterable[Iterable[str]]
Path = Union[_Path, str]
Sequence = List[Set[str]]
