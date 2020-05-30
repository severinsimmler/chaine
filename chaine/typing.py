import numpy as np
from pathlib import Path
from typing import Union, Callable, List, Generator, Optional

Filepath = Union[str, Path]
Vector = Union[List[str], np.ndarray]
Matrix = Union[List[Vector], np.ndarray]
MatrixGenerator = Generator[Matrix, None, None]
SentenceGenerator = Generator["Sentence", None, None]
FeatureGenerator = Generator[List[str], None, None]
TokenGenerator = Generator["Token", None, None]