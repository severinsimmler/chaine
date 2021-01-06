# Train a conditional random field for named entity recognition

Introduction


## Setup


```
$ pip install datasets seqeval
```

If you have cloned this repository, simply run:

```
$ poetry install
```


## How it works

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
