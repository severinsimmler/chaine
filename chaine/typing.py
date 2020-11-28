from typing import List, Optional, Generator

FeatureGenerator = Generator[List[str], None, None]
TokenGenerator = Generator["Token", None, None]
