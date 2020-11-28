# A lightweight Linear-Chain Conditional Random Field

This is a modern, fast and no-dependency Python library implementing a linear-chain conditional random field for natural language processing tasks like named entity recognition or part-of-speech tagging.


## Installation

You can install the latest stable version from [PyPI](https://pypi.org/project/chaine):

```
$ pip install chaine
```


## Example

```python
>>> import chaine
>>> import datasets
>>> data = datasets.load_dataset("germeval_14")
>>> tokens = data["train"]["tokens"]
>>> labels = data["train"]["ner_tags"]
>>> crf = chaine.train(tokens, labels, max_iterations=100)
>>> sequence = chaine.featurize(["todo", "todo", "todo"])
>>> crf.predict(sequence)
["O", "O", "B-PER"]
```


## Disclaimer

This library makes use of and is partially based on:

- [CRFsuite](https://github.com/chokkan/crfsuite)
- [libLBFGS](https://github.com/chokkan/liblbfgs)
- [python-crfsuite](https://github.com/scrapinghub/python-crfsuite)
- [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite)
