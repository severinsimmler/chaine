# Chaine

A linear-chain conditional random field implementation.

Chaine is a modern Python library without any third-party dependencies and a backend written in C implementing conditional random fields for natural language processing tasks like named entity recognition or part-of-speech tagging.

- **Lightweight:** explain
- **Fast:** explain
- **Easy to use:** explain

You can install the latest stable version from [PyPI](https://pypi.org/project/chaine):

```
$ pip install chaine
```

If you are interested in the theoretical concepts behind conditional random fields, refer to the introducing paper by [Lafferty et al](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers).


## How it works

```python
>>> import chaine
>>> tokens = [["John", "Lennon", "was", "rhythm", "guitarist" "of", "The", "Beatles"]]
>>> labels = [["B-PER", "I-PER", "O", "O", "O", "O", "B-ORG", "I-ORG"]]
>>> model = chaine.train(tokens, labels, max_iterations=5)
Loading data
Start training
Iteration 1, train loss: 14.334076
Iteration 2, train loss: 14.334064
Iteration 3, train loss: 14.334053
Iteration 4, train loss: 14.334041
Iteration 5, train loss: 14.334029
>>> model.predict(tokens)
[['B-PER', 'I-PER', 'O', 'O', 'O', 'B-ORG', 'I-ORG']]
```

Check out the introducing [Jupyter notebook](https://github.com/severinsimmler/chaine/blob/master/notebooks/tutorial.ipynb).


## Credits

This library makes use of and is partially based on:

- [CRFsuite](https://github.com/chokkan/crfsuite)
- [libLBFGS](https://github.com/chokkan/liblbfgs)
- [python-crfsuite](https://github.com/scrapinghub/python-crfsuite)
- [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite)
