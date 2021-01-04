# Chaine

Linear-chain conditional random fields for natural language processing.

Chaine is a modern Python library without third-party dependencies and a backend written in C. You can train conditional random fields for natural language processing tasks like [named entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) or [part-of-speech tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging).

- **Lightweight**: No use of bloated third-party libraries — only pure Python and C.
- **Fast**: Performance critical parts are written in C and thus [blazingly fast](http://www.chokkan.org/software/crfsuite/benchmark.html).
- **Easy to use**: Designed with special focus on usability and a beautiful high-level API.

You can install the latest stable version from [PyPI](https://pypi.org/project/chaine):

```
$ pip install chaine
```

If you are interested in the theoretical concepts behind conditional random fields, please refer to the introducing paper by [Lafferty et al](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers).


## Example

```python
>>> import chaine
>>> tokens = [["John", "Lennon", "was", "born", "in" "Liverpool"]]
>>> labels = [["B-PER", "I-PER", "O", "O", "O", "B-LOC"]]
>>> model = chaine.train(tokens, labels, max_iterations=5)
>>> model.predict(tokens)
[['B-PER', 'I-PER', 'O', 'O', 'O', 'B-LOC']]
```

Check out the introducing [Jupyter notebook](https://github.com/severinsimmler/chaine/blob/master/notebooks/tutorial.ipynb).


## Credits

This library makes use of and is partially based on:

- [CRFsuite](https://github.com/chokkan/crfsuite)
- [libLBFGS](https://github.com/chokkan/liblbfgs)
- [python-crfsuite](https://github.com/scrapinghub/python-crfsuite)
- [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite)
