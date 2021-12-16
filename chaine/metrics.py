from collections import Counter
from typing import Iterable

def foo(true: Iterable[str], pred: Iterable[str]) -> dict[str, int]:
    counts = Counter()

    true = [l.removeprefix("B-").removeprefix("I-") for l in true]
    pred = [l.removeprefix("B-").removeprefix("I-") for l in pred]

def precision(true: Iterable[str], pred: Iterable[str]) -> float:
    counts = Counter()

    true = [l.removeprefix("B-").removeprefix("I-") for l in true]
    pred = [l.removeprefix("B-").removeprefix("I-") for l in pred]

    for t, p in zip(true, pred):
        if t != "O" and t == p:
            counts.update(["tp"])
        elif t == "O" and t != p:
            counts.update(["fp"])

    return (counts["tp"] + counts["tn"]) / (counts["tp"] + counts["fp"] + counts["tn"] + counts["fn"])


def recall():
    pass


def f1():
    pass


def evaluate_model():
    pass
