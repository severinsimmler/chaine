"""
chaine.optimization.metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements metrics to evaluate the performance of a trained model.
"""

from collections import Counter


def calculate_precision(true_positives: int, false_positives: int) -> float:
    """Calculate precision score.

    Parameters
    ----------
    true_positives : int
        Number of true positives.
    false_positives : int
        Number of false positives.

    Returns
    -------
    float
        Precision score.
    """
    try:
        return true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        # only false negatives is perfect precision
        return 1.0


def calculate_recall(true_positives: int, false_negatives: int) -> float:
    """Calculate recall score.

    Parameters
    ----------
    true_positives : int
        Number of true positives.
    false_negatives : int
        Number of false negatives.

    Returns
    -------
    float
        Recall score.
    """
    try:
        return true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        # only false positives is imperfect recall
        return 0.0


def calculate_f1(true_positives: int, false_positives: int, false_negatives: int) -> float:
    """Calculate F1 score.

    Parameters
    ----------
    true_positives : int
        Number of true positives.
    false_negatives : int
        Number of false negatives.

    Returns
    -------
    float
        Precision score
    """
    precision = calculate_precision(true_positives, false_positives)
    recall = calculate_recall(true_positives, false_negatives)
    try:
        return (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        # zero precision and zero recall is zero f1
        return 0.0


def evaluate_predictions(true: list[list[str]], pred: list[list[str]]) -> dict[str, float]:
    """Evaluate the given predictions with the true labels.

    Parameters
    ----------
    true : list[list[str]]
        True labels.
    pred : list[list[str]]
        Predicted labels.

    Returns
    -------
    dict[str, float]
        Precision, recall and F1 scores.
    """
    # validate input
    if (
        not isinstance(true, list)
        or not isinstance(pred, list)
        or not isinstance(true[0], list)
        or not isinstance(pred[0], list)
    ):
        raise ValueError("Input lists are invalid")

    counts = Counter()

    # get true positives, true negatives, false positives, false negatives
    for true_labels, predicted_labels in zip(true, pred):
        # ignore prefixes
        true_labels = [l.removeprefix("B-").removeprefix("I-") for l in true_labels]
        predicted_labels = [l.removeprefix("B-").removeprefix("I-") for l in predicted_labels]

        if len(true_labels) != len(predicted_labels):
            raise ValueError(f"Different lengths: '{true_labels}' vs. '{predicted_labels}'")

        for true_label, predicted_label in zip(true_labels, predicted_labels):
            if true_label != "O" and predicted_label == true_label:
                counts["tp"] += 1
            if predicted_label != "O" and predicted_label != true_label:
                counts["fp"] += 1
            if true_label == "O" and predicted_label == "O":
                counts["tn"] += 1
            if true_label != "O" and predicted_label == "O":
                counts["fn"] += 1

    # calculate precision, recall and f1 score
    return {
        "precision": calculate_precision(counts["tp"], counts["fp"]),
        "recall": calculate_recall(counts["tp"], counts["fn"]),
        "f1": calculate_f1(counts["tp"], counts["fp"], counts["fn"]),
    }
