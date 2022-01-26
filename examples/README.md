# Overview

1. [Setup](#setup)
2. [Train a conditional random field for named entity recognition](#train-a-conditional-random-field-for-named-entity-recognition)
3. [Optimize hyperparameters of a conditional random field](#optimize-hyperparameters-of-a-conditional-random-field)

## Setup

To run the example scripts, make sure you have installed `chaine` and two additional dependencies (to load a dataset and evaluate the model):

```
$ pip install chaine datasets seqeval
```

> These two additional dependencies are obviously optional, i.e. you don't need them to use `chaine` itself, but to run the example scripts.

## Train a conditional random field for named entity recognition

Named entity recognition (NER) seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, etc.

This can be considered as a sequence labeling problem. Given a sequence of tokens:

```python
["John", "Lennon", "was", "born", "in", "Liverpool"]
```

the goal is to find the most likely sequence of named entity labels:

```python
["B-PER", "I-PER", "O", "O", "O", "B-LOC"]
```

One approach to solve this problem is to train a linear-chain conditional random field (CRF). Unlike a [hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) (HMM; a different but directed model for sequential data), a CRF can access the complete information of the input sequence at any point, whereas an HMM sees only the current input. This allows complex feature sets to be used.

### Start training

You can train and evaluate a CRF with the English [CoNLL 2003](https://www.clips.uantwerpen.be/conll2003/ner/) data set:

```sh
$ python ner.py
```

This should only take a few minutes, serializes the trained model and outputs a classification report with precision, recall and f1 scores.

### How it works

Please refer to the excellent [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) by Dan Jurafsky and James H. Martin, especially the chapter [Sequence Labeling for Parts of Speech and Named Entities](https://web.stanford.edu/~jurafsky/slp3/8.pdf) for a general introduction.

Features for a token are represented with `chaine` as dictionary. For example, features for the token `John` might be:

```python
{
    "text": "john",
    "is_capitalized": True,
    "part_of_speech": "NN",
}
```

They key of a feature dictionary must be a string, values may be strings, booleans, integers or floats.

Including certain features from the preceding or following tokens can have quite a positive effect. In [`featurize_token()`](ner.py), we explicitly model whether a token is at the beginning or end of a sequence and which tokens occur before or after it.

## Optimize hyperparameters of a conditional random field

Quoting [Wikipedia](https://en.wikipedia.org/wiki/Hyperparameter_optimization):

> In machine learning, hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm. A hyperparameter is a parameter whose value is used to control the learning process. By contrast, the values of other parameters (typically node weights) are learned.
>
> The same kind of machine learning model can require different constraints, weights or learning rates to generalize different data patterns. These measures are called hyperparameters, and have to be tuned so that the model can optimally solve the machine learning problem.

You can tune hyperparameters with `chaine` using random search, i.e. you define a search space (a range of numbers or values) and `chaine` will randomly choose from the defined search space. A k-fold cross-validation is performed by default to estimate the generalization performance of the model.

Since this can be a quite time-consuming and computationally expensive process, you can also downsample the data set.

### Start optimization

Running the example script will use a downsampled version of the English [CoNLL 2003](https://www.clips.uantwerpen.be/conll2003/ner/) data set with 5-fold cross-validation and search spaces for all available algorithms:

```sh
$ python optimization.py
```

This should take a few minutes and write a JSON with the ranked results (hyperparameter settings and evaluation scores).

### How it works

For each training method, you can define a search space, e.g.:

```python
>>> from chaine.optimization.spaces import L2SGDSearchSpace
>>> from chaine.optimization.utils import NumberSeries
>>> space = L2SGDSearchSpace(
...    min_freq=NumberSeries(start=0, stop=5, step=1),
...    all_possible_states={True, False}
...)
```

and pass this search space with the data set to the optimization trial:

```python
>>> from chaine.optimization.trial import OptimizationTrial
>>> with OptimizationTrial(dataset, space) as trial:
...     print(f"Result: {trial}")
```
