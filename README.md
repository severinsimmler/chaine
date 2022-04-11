# Chaine

[![downloads](https://static.pepy.tech/personalized-badge/chaine?period=total&units=international_system&left_color=black&right_color=black&left_text=downloads)](https://pepy.tech/project/chaine)
[![downloads/month](https://static.pepy.tech/personalized-badge/chaine?period=month&units=abbreviation&left_color=black&right_color=black&left_text=downloads/month)](https://pepy.tech/project/chaine)
[![downloads/week](https://static.pepy.tech/personalized-badge/chaine?period=week&units=abbreviation&left_color=black&right_color=black&left_text=downloads/week)](https://pepy.tech/project/chaine)

Chaine is a modern, fast and lightweight Python library implementing linear-chain conditional random fields. Use it for sequence labeling tasks like [named entity recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) or [part-of-speech tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging).

The main goals of this project are:

- **Usability**: Designed with special focus on usability and a beautiful high-level API.
- **Efficiency**: Performance critical parts are written in C and thus [blazingly fast](http://www.chokkan.org/software/crfsuite/benchmark.html). Loading a model from disk and retrieving feature weights for inference is optimized for both [speed and memory](http://www.chokkan.org/software/cqdb/).
- **Persistency**: No `pickle`, `joblib` etc. is used for serialization.  A trained model will be compatible with all versions for eternity, because the underlying C library will not change. I promise.
- **Minimalism**: No code bloat, no external dependencies.

Install the latest stable version from [PyPI](https://pypi.org/project/chaine):

```
pip install chaine
```

## Algorithms

You can train models using the following methods:

- Limited-Memory BFGS ([Nocedal 1980](https://www.jstor.org/stable/2006193))
- Orthant-Wise Limited-Memory Quasi-Newton ([Andrew et al. 2007](https://www.microsoft.com/en-us/research/publication/scalable-training-of-l1-regularized-log-linear-models/))
- Stochastic Gradient Descent ([Shalev et al. 2007](https://www.google.com/url?q=https://www.cs.huji.ac.il/~shais/papers/ShalevSiSr07.pdf))
- Averaged Perceptron ([Collins 2002](https://aclanthology.org/W02-1001.pdf))
- Passive Aggressive ([Crammer et al. 2006](https://jmlr.csail.mit.edu/papers/v7/crammer06a.html))
- Adaptive Regularization of Weight Vectors ([Mejer et al. 2010](https://aclanthology.org/D10-1095.pdf))

Please refer to the paper by [Lafferty et al.](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers) for a general introduction to conditional random fields.

## Usage

Training and using a conditional random field for inference is easy as:

```python
>>> import chaine
>>> tokens = [[{"index": 0, "text": "John"}, {"index": 1, "text": "Lennon"}]]
>>> labels = [["B-PER", "I-PER"]]
>>> model = chaine.train(tokens, labels)
>>> model.predict(tokens)
[['B-PER', 'I-PER']]
```

### Features

One token in a sequence is represented as a dictionary with describing feature names as keys and respective values of type string, integer, float or boolean:

```python
{
    "text": "John",
    "num_characters": 4,
    "relative_index": 0.0,
    "is_number": False,
}
```

One sequence is represented as a list of feature dictionaries:

```python
[{"text": "John"}, {"text": "Lennon"}]
```

One data set is represented as an iterable of a list of feature dictionaries:

```python
[[{"text": "John"}, {"text": "Lennon"}]]
```

This is the expected input format for training. For inference, you can also process a single sequence rather than a batch of multiple sequences.

Depending on the size of your data set, it probably makes sense to use generators. Something like this would be totally fine for both training and inference:

```python
([extract_features(token) for token in tokens] for tokens in dataset)
```

Assuming `dataset` is a generator as well, only one sequence is loaded into memory at a time.

### Training

You can either use the high-level function to train a model (which also loads and returns it):

```python
>>> import chaine
>>> chaine.train(tokens, labels)
```

or the lower-level `Trainer` class:

```python
>>> from chaine import Trainer
>>> trainer = Trainer()
```

A `Trainer` object has a method `train()` to learn states and transitions from the given data set. You have to provide a filepath to serialize the model to:

```python
>>> trainer.train(tokens, labels, model_filepath="model.chaine")
```

### Hyperparameters

Before training a model, you might want to find out the ideal hyperparameters first. You can just set the respective argument to `True`:

```python
>>> import chaine
>>> model = chaine.train(tokens, labels, optimize_hyperparameters=True)
```

> This might be very memory and time consuming, because 5-fold cross validation for each of the 10 trials for each of the algorithms is performed.

or use the `HyperparameterOptimizer` class and have more control over the optimization process:

```python
>>> from chaine import HyperparameterOptimizer
>>> from chaine.optimization import L2SGDSearchSpace
>>> optimizer = HyperparameterOptimizer(trials=50, folds=3, spaces=[L2SGDSearchSpace()])
>>> optimizer.optimize_hyperparameters(tokens, labels, sample_size=1000)
```

This will make 50 trails with 3-fold cross validation for the Stochastic Gradient Descent algorithm and return a sorted list of hyperparameters with evaluation stats. The given data set is downsampled to 1000 instances.

<details>
<summary>Example of a hyperparameter optimization report</summary>

```json
[
    {
        "hyperparameters": {
            "algorithm": "lbfgs",
            "min_freq": 0,
            "all_possible_states": true,
            "all_possible_transitions": true,
            "num_memories": 8,
            "c1": 0.9,
            "c2": 0.31,
            "epsilon": 0.00011,
            "period": 17,
            "delta": 0.00051,
            "linesearch": "Backtracking",
            "max_linesearch": 31
        },
        "stats": {
            "mean_precision": 0.4490952380952381,
            "stdev_precision": 0.16497993418839532,
            "mean_recall": 0.4554858934169279,
            "stdev_recall": 0.20082402876210334,
            "mean_f1": 0.45041435392087253,
            "stdev_f1": 0.17914435056760908,
            "mean_time": 0.3920876979827881,
            "stdev_time": 0.0390961164333519
        }
    },
    {
        "hyperparameters": {
            "algorithm": "lbfgs",
            "min_freq": 5,
            "all_possible_states": true,
            "all_possible_transitions": false,
            "num_memories": 9,
            "c1": 1.74,
            "c2": 0.09,
            "epsilon": 0.0008600000000000001,
            "period": 1,
            "delta": 0.00045000000000000004,
            "linesearch": "StrongBacktracking",
            "max_linesearch": 34
        },
        "stats": {
            "mean_precision": 0.4344436335328176,
            "stdev_precision": 0.15542689556199216,
            "mean_recall": 0.4385174258109041,
            "stdev_recall": 0.19873733310765845,
            "mean_f1": 0.43386496201052716,
            "stdev_f1": 0.17225578421967264,
            "mean_time": 0.12209572792053222,
            "stdev_time": 0.0236177196325414
        }
    },
    {
        "hyperparameters": {
            "algorithm": "lbfgs",
            "min_freq": 2,
            "all_possible_states": true,
            "all_possible_transitions": true,
            "num_memories": 1,
            "c1": 0.91,
            "c2": 0.4,
            "epsilon": 0.0008400000000000001,
            "period": 13,
            "delta": 0.00018,
            "linesearch": "MoreThuente",
            "max_linesearch": 43
        },
        "stats": {
            "mean_precision": 0.41963433149859447,
            "stdev_precision": 0.16363544501259455,
            "mean_recall": 0.4331173486012196,
            "stdev_recall": 0.21344965207006913,
            "mean_f1": 0.422038027332145,
            "stdev_f1": 0.18245844823319127,
            "mean_time": 0.2586916446685791,
            "stdev_time": 0.04341208573100539
        }
    },
    {
        "hyperparameters": {
            "algorithm": "l2sgd",
            "min_freq": 5,
            "all_possible_states": true,
            "all_possible_transitions": true,
            "c2": 1.68,
            "period": 2,
            "delta": 0.00047000000000000004,
            "calibration_eta": 0.0006900000000000001,
            "calibration_rate": 2.9000000000000004,
            "calibration_samples": 1400,
            "calibration_candidates": 25,
            "calibration_max_trials": 23
        },
        "stats": {
            "mean_precision": 0.2571428571428571,
            "stdev_precision": 0.43330716823151716,
            "mean_recall": 0.01,
            "stdev_recall": 0.022360679774997897,
            "mean_f1": 0.01702127659574468,
            "stdev_f1": 0.038060731531911314,
            "mean_time": 0.15442829132080077,
            "stdev_time": 0.051750737506044905
        }
    }
]
```
</details>

### Inference

The high-level function `chaine.train()` returns a `Model` object. You can load an already trained model from disk by initializing a `Model` object with the model's filepath:

```python
>>> from chaine import Model
>>> model = Model("model.chaine")
```

You can predict labels for a batch of sequences:

```python
>>> tokens = [
...     [{"index": 0, "text": "John"}, {"index": 1, "text": "Lennon"}],
...     [{"index": 0, "text": "Paul"}, {"index": 1, "text": "McCartney"}],
...     [{"index": 0, "text": "George"}, {"index": 1, "text": "Harrison"}],
...     [{"index": 0, "text": "Ringo"}, {"index": 1, "text": "Starr"}]
... ]
>>> model.predict(tokens)
[['B-PER', 'I-PER'], ['B-PER', 'I-PER'], ['B-PER', 'I-PER'], ['B-PER', 'I-PER']]
```

or only for a single sequence:

```python
>>> model.predict_single(tokens[0])
['B-PER', 'I-PER']
```

If you are interested in the model's probability distribution for a given sequence, you can:

```python
>>> model.predict_proba_single(tokens[0])
[[{'B-PER': 0.99, 'I-PER': 0.01}, {'B-PER': 0.01, 'I-PER': 0.99}]]
```

> Use the `model.predict_proba()` method for a batch of sequences.

### Weights

After loading a trained model, you can inspect the learned transition and state weights:

```python
>>> model = Model("model.chaine")
>>> model.transitions
[{'from': 'B-PER', 'to': 'I-PER', 'weight': 1.430506540616852e-06}]
>>> model.states
[{'feature': 'text:John', 'label': 'B-PER', 'weight': 9.536710877105517e-07}, ...]
```

You can also dump both transition and state weights as JSON:

```python
>>> model.dump_states("states.json")
>>> model.dump_transitions("transitions.json")
```

## Credits

This project makes use of and is partially based on:

- [CRFsuite](https://github.com/chokkan/crfsuite)
- [libLBFGS](https://github.com/chokkan/liblbfgs)
- [python-crfsuite](https://github.com/scrapinghub/python-crfsuite)
- [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite)
