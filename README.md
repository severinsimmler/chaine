# A lightweight Linear-Chain Conditional Random Field

This is a work in progress.

## Installation

You can install the latest stable version from [PyPI](https://pypi.org/project/chaine):

```
$ pip install chaine
```

## Example

```python
>>> import chaine
>>> tokens = ["Foo", "bar"]
>>> matrix = chaine.matrix(tokens)
>>> matrix
<FeatureMatrix: 2 Tokens>
>>> matrix.numpy()
array([[0, 1, 2, 3],
       [4, 1, 5, 3]])
```
