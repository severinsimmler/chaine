# Train a conditional random field for named entity recognition

TODO: Introduction


## Setup

Make sure you have installed `chaine` and some additional dependencies (to load a dataset and evaluate the model):

```
$ pip install datasets seqeval
```


## How it works

TODO: refer to Jurafsky and Martin


```
["John", "Lennon"]
```

becomes

```
{"text": "john", "is_capitalized": True}
```

becomes

```
{"text": "john", "is_capitalized": True, "text+1": "lennon", "is_capitalized+1": True}
```
