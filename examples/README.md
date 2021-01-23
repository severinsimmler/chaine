# Train a conditional random field for named entity recognition

Named entity recognition (NER) seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, etc.

This can be considered as a sequence labeling problem. Given a sequence of tokens:

```
["John", "Lennon", "was", "born", "in", "Liverpool"]
```

the goal is to find the most likely sequence of named entity labels:

```
["B-PER", "I-PER", "O", "O", "O", "B-LOC"]
```

One approach to solve this problem is to train a linear-chain conditional random field (CRF). Unlike a [hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) (HMM; a different but directed model for sequential data), a CRF can access the complete information of the input sequence at any point, whereas an HMM sees only the current input. This allows complex feature sets to be used.


## Setup

Make sure you have installed `chaine` and two additional dependencies (to load a dataset and evaluate the model):

```sh
$ pip install datasets seqeval
```

## Start training

You can train and evaluate a CRF with the English [CoNLL 2003](https://www.clips.uantwerpen.be/conll2003/ner/) data set:

```sh
python ner.py
```

This should only take a few minutes, serializes the trained model and outputs a classification report with precision, recall and f1 scores.


## How it works

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

Including certain features from the preceding or following tokens can have quite a positive effect. In [`featurize_token()`](), we explicitly model whether a token is at the beginning or end of a sequence and which tokens occur before or after it.
