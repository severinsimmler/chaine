from typing import List, Optional, Generator, Iterable

FeatureGenerator = Generator[List[str], None, None]
TokenGenerator = Generator["Token", None, None]
Labels = Iterable[str]
