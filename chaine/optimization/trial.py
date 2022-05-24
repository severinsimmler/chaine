"""
chaine.optimization.trial
~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements a class for a hyperparameter optimization trial.
"""

import statistics
import tempfile
import time
import uuid
from pathlib import Path

from chaine.optimization.metrics import evaluate_predictions
from chaine.optimization.spaces import SearchSpace
from chaine.typing import Iterable, Iterator, Labels, Sequence


class OptimizationTrial:
    def __init__(
        self,
        splits: Iterator[tuple[tuple[Iterable[Sequence], Iterable[Labels]]]],
        space: SearchSpace,
        *,
        is_baseline: bool
    ):
        """Hyperparameter optimization trial.

        Parameters
        ----------
        splits : Iterator[tuple[tuple[Iterable[Sequence], Iterable[Labels]]]]
            K-fold split data set.
        space : SearchSpace
            Search space for hyperparameter optimization.
        is_baseline : bool
            True if trial is a baseline (i.e. default hyperparameters to be used).
        """
        self.splits = splits
        self.space = space
        self.is_baseline = is_baseline
        self.model_filepath = Path(tempfile.gettempdir(), str(uuid.uuid4()))
        self.precision = []
        self.recall = []
        self.f1 = []
        self.time = []

    def __enter__(self) -> dict[str, dict]:
        """Train and evaluate a model.

        Returns
        -------
        dict[str, dict]
            Selected hyperparameters and evaluation scores.
        """
        from chaine.crf import Model, Trainer

        if self.is_baseline:
            # default hyperparameters as baseline
            params = {"algorithm": self.space.algorithm}
        else:
            # select random hyperparameters
            params = self.space.random_hyperparameters()

        for (train_dataset, train_labels), (test_dataset, test_labels) in self.splits:
            # fire!
            start = time.time()
            trainer = Trainer(max_iterations=100, **params)
            trainer.train(train_dataset, train_labels, model_filepath=self.model_filepath)
            end = time.time()

            # evaluate model
            model = Model(self.model_filepath)
            predicted_labels = model.predict(test_dataset)
            scores = evaluate_predictions(test_labels, predicted_labels)

            # save scores
            self.precision.append(scores["precision"])
            self.recall.append(scores["recall"])
            self.f1.append(scores["f1"])
            self.time.append(end - start)

        # return both hyperparameters and evaluation metrics
        return {
            "hyperparameters": params,
            "stats": {
                "mean_precision": statistics.mean(self.precision) if self.precision else None,
                "stdev_precision": statistics.stdev(self.precision) if self.precision else None,
                "mean_recall": statistics.mean(self.recall) if self.recall else None,
                "stdev_recall": statistics.stdev(self.recall) if self.recall else None,
                "mean_f1": statistics.mean(self.f1) if self.f1 else None,
                "stdev_f1": statistics.stdev(self.f1) if self.f1 else None,
                "mean_time": statistics.mean(self.time) if self.time else None,
                "stdev_time": statistics.stdev(self.time) if self.time else None,
            },
        }

    def __exit__(self, *args) -> bool:
        # clean up
        if self.model_filepath.exists():
            self.model_filepath.unlink()

        # ignore exceptions
        return True
