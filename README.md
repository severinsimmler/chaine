# A lightweight Linear-Chain Conditional Random Field implementation

Introduction


## Installation

You can install the latest stable version from [PyPI](https://pypi.org/project/chaine):

```
$ pip install chaine
```

## Example

```python
>>> from chaine import ConditionalRandomField
>>> crf = ConditionalRandomField()
>>> crf.train(dataset)
>>> crf.predict(dataset)
```
