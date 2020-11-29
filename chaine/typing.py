"""
chaine.typing
~~~~~~~~~~~~~

A collection of type hints
"""

from typing import List, Optional, Generator, Iterable, Union, Any

FeatureGenerator = Generator[List[str], None, None]
TokenGenerator = Generator["Token", None, None]
Labels = Iterable[str]
