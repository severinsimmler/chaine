from pathlib import Path
from typing import Union, Callable, List, Generator, Optional, Iterable

Filepath = Union[str, Path]
Vector = Union[List[str]]
Matrix = Union[List[Vector]]
MatrixGenerator = Generator[Matrix, None, None]
SentenceGenerator = Generator["Sentence", None, None]
FeatureGenerator = Generator[List[str], None, None]
TokenGenerator = Generator["Token", None, None]
FeatureGenerator = Generator[List[int], None, None]
