import pytest

from chaine.optimization import metrics


def test_calculate_precision():
    assert metrics.calculate_precision(10, 5) == 0.6666666666666666
    assert metrics.calculate_precision(5, 10) == 0.3333333333333333
    assert metrics.calculate_precision(99, 1) == 0.99


def test_calculate_recall():
    assert metrics.calculate_recall(10, 5) == 0.6666666666666666
    assert metrics.calculate_recall(5, 10) == 0.3333333333333333
    assert metrics.calculate_recall(99, 1) == 0.99


def test_calculate_f1():
    assert metrics.calculate_f1(1, 2, 3) == 0.28571428571428575
    assert metrics.calculate_f1(4, 5, 6) == 0.4210526315789474
    assert metrics.calculate_f1(1000, 50, 12) == 0.9699321047526673


def test_calculate_evaluate_predictions():
    true = [["B-A", "I-A", "I-A", "O", "B-B", "I-B"]]
    pred = [["B-A", "I-A", "I-A", "O", "B-B", "I-B"]]
    assert metrics.evaluate_predictions(true, pred) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    true = [["B-A", "I-A", "I-A", "O", "B-B", "I-B"]]
    pred = [["B-A", "I-A", "I-A", "B-B", "I-B", "I-B"]]
    assert metrics.evaluate_predictions(true, pred) == {
        "precision": 0.8333333333333334,
        "recall": 1.0,
        "f1": 0.9090909090909091,
    }

    true = [["B-A", "I-A", "I-A", "O", "B-B", "I-B"]]
    pred = [["B-A", "I-A", "I-A", "O", "O", "O"]]
    assert metrics.evaluate_predictions(true, pred) == {
        "precision": 1.0,
        "recall": 0.6,
        "f1": 0.7499999999999999,
    }

    true = [["B-A", "I-A", "I-A", "O", "B-B", "I-B"]]
    pred = [["B-A", "B-A", "B-A", "O", "I-B", "I-B"]]
    assert metrics.evaluate_predictions(true, pred) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    true = [["B-A", "I-A", "I-A", "O", "B-B", "I-B"]]
    pred = [["B-A", "O", "O", "O", "O", "O"]]
    assert metrics.evaluate_predictions(true, pred) == {
        "precision": 1.0,
        "recall": 0.2,
        "f1": 0.33333333333333337,
    }

    true = [["B-A", "I-A", "I-A", "O", "O", "O"]]
    pred = [["B-A", "I-A", "I-A", "B-B", "I-B", "I-B"]]
    assert metrics.evaluate_predictions(true, pred) == {
        "precision": 0.5,
        "recall": 1.0,
        "f1": 0.6666666666666666,
    }

    with pytest.raises(ValueError):
        true = ["B-A", "I-A", "I-A", "O", "B-B", "I-B"]
        pred = ["B-A", "I-A", "I-A", "O", "B-B", "I-B"]
        metrics.evaluate_predictions(true, pred)
