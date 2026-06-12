"""
chaine.optimization.trial
~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements a class for a hyperparameter optimization trial.
"""

import statistics
import tempfile
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

from chaine.optimization.metrics import evaluate_predictions
from chaine.optimization.spaces import SearchSpace
from chaine.optimization.utils import Fold


class OptimizationTrial:
    def __init__(
        self,
        splits: Iterable[tuple[Fold, Fold]],
        space: SearchSpace,
        *,
        is_baseline: bool,
    ):
        """Hyperparameter optimization trial.

        Parameters
        ----------
        splits : Iterable[tuple[Fold, Fold]]
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

    def run(self) -> dict[str, dict]:
        """Train and evaluate a model on every split.

        Returns
        -------
        dict[str, dict]
            Selected hyperparameters and evaluation scores.
        """
        # late import to avoid circular dependency
        from chaine.crf import Model, Trainer

        if self.is_baseline:
            # default hyperparameters as baseline
            params = {"algorithm": self.space.algorithm}
        else:
            # select random hyperparameters
            params = self.space.random_hyperparameters()

        try:
            for (train_dataset, train_labels), (test_dataset, test_labels) in self.splits:
                # fire!
                start = time.perf_counter()
                trainer = Trainer(max_iterations=100, **params)
                trainer.train(train_dataset, train_labels, model_filepath=self.model_filepath)
                self.time.append(time.perf_counter() - start)

                # evaluate model
                model = Model(self.model_filepath)
                predicted_labels = model.predict(test_dataset)
                scores = evaluate_predictions(test_labels, predicted_labels)

                # save scores
                self.precision.append(scores["precision"])
                self.recall.append(scores["recall"])
                self.f1.append(scores["f1"])
        finally:
            # clean up
            self.model_filepath.unlink(missing_ok=True)

        # return both hyperparameters and evaluation metrics
        return {
            "hyperparameters": params,
            "stats": {
                **self._stats("precision", self.precision),
                **self._stats("recall", self.recall),
                **self._stats("f1", self.f1),
                **self._stats("time", self.time),
            },
        }

    @staticmethod
    def _stats(name: str, values: list[float]) -> dict[str, float | None]:
        """Mean and standard deviation of the given values (or None if no values)."""
        return {
            f"mean_{name}": statistics.mean(values) if values else None,
            f"stdev_{name}": statistics.stdev(values) if values else None,
        }
