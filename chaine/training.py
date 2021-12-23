"""
chaine.api
~~~~~~~~~~

This module implements the high-level API to train a conditional random field.
"""

import random
import statistics
import time
from operator import itemgetter
from typing import Callable, Optional, Union

from chaine.crf import Model, Trainer
from chaine.logging import Logger
from chaine.optimization import (
    APSearchSpace,
    AROWSearchSpace,
    L2SGDSearchSpace,
    LBFGSSearchSpace,
    PASearchSpace,
    SearchSpace,
)
from chaine.optimization.metrics import evaluate
from chaine.optimization.utils import cross_validation
from chaine.typing import Filepath, Iterable, Labels, Sequence

LOGGER = Logger(__name__)


def train(
    dataset: Iterable[Sequence],
    labels: Iterable[Labels],
    model_filepath: Filepath = "model.crf",
    **kwargs,
) -> Model:
    """Train a conditional random field.

    Parameters
    ----------
    dataset : Iterable[Sequence]
        Data set consisting of sequences of feature sets.
    labels : Iterable[Labels]
        Labels corresponding to each instance in the data set.
    model_filepath : Filepath, optional (default=model.crf)
        Path to model location.
    algorithm : str
        The following optimization algorithms are available:
            * lbfgs: Limited-memory BFGS with L1/L2 regularization
            * l2sgd: Stochastic gradient descent with L2 regularization
            * ap: Averaged perceptron
            * pa: Passive aggressive
            * arow: Adaptive regularization of weights

    Limited-memory BFGS Parameters (lbfgs)
    --------------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (unlimited by default).
    num_memories : int, optional (default=6)
        Number of limited memories for approximating the inverse hessian matrix.
    c1 : float, optional (default=0)
        Coefficient for L1 regularization.
    c2 : float, optional (default=1.0)
        Coefficient for L2 regularization.
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence.
    period : int, optional (default=10)
        Threshold value for iterations to test the stopping criterion.
    delta : float, optional (default=1e-5)
        Top iteration when log likelihood is not greater than this.
    linesearch : str, optional (default="MoreThuente")
        Line search algorithm used in updates:
            * MoreThuente: More and Thuente's method
            * Backtracking: Backtracking method with regular Wolfe condition
            * StrongBacktracking: Backtracking method with strong Wolfe condition
    max_linesearch : int, optional (default=20)
        Maximum number of trials for the line search algorithm.

    SGD with L2 Parameters (l2sgd)
    ------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (1000 by default).
    c2 : float, optional (default=1.0)
        Coefficient for L2 regularization.
    period : int, optional (default=10)
        Threshold value for iterations to test the stopping criterion.
    delta : float, optional (default=1e-5)
        Top iteration when log likelihood is not greater than this.
    calibration_eta : float, optional (default=0.1)
        Initial value of learning rate (eta) used for calibration.
    calibration_rate : float, optional (default=2.0)
        Rate of increase/decrease of learning rate for calibration.
    calibration_samples : int, optional (default=1000)
        Number of instances used for calibration.
    calibration_candidates : int, optional (default=10)
        Number of candidates of learning rate.
    calibration_max_trials : int, optional (default=20)
        Maximum number of trials of learning rates for calibration.

    Averaged Perceptron Parameters (ap)
    -----------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default).
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence.

    Passive Aggressive Parameters (pa)
    ----------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default).
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence.
    pa_type : int, optional (default=1)
        Strategy for updating feature weights:
            * 0: PA without slack variables
            * 1: PA type I
            * 2: PA type II
    c : float, optional (default=1)
        Aggressiveness parameter (used only for PA-I and PA-II).
    error_sensitive : bool, optional (default=True)
        Include square root of predicted incorrect labels into optimization routine.
    averaging : bool, optional (default=True)
        Compute average of feature weights at all updates.

    Adaptive Regularization of Weights Parameters (arow)
    ----------------------------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default).
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence.
    variance : float, optional (default=1)
        Initial variance of every feature weight.
    gamma : float, optional (default=1)
        Trade-off between loss function and changes of feature weights.

    Returns
    -------
    Model
        A conditional random field trained on the dataset.
    """
    # initialize trainer and start training
    trainer = Trainer(**kwargs)
    trainer.train(dataset, labels, model_filepath=str(model_filepath))

    # load and return the trained model
    return Model(model_filepath)


def optimize(
    dataset: Iterable[Sequence],
    labels: Iterable[Labels],
    spaces: list[SearchSpace] = [
        AROWSearchSpace(),
        APSearchSpace(),
        LBFGSSearchSpace(),
        L2SGDSearchSpace(),
        PASearchSpace(),
    ],
    metric: str = "f1",
    trials: int = 10,
    cv: int = 5,
    seed: Optional[int] = None,
) -> dict[str, Union[str, int, float, bool]]:
    """Optimize hyperparameters in a randomized manner on the given data set.

    For each hyperparameter search space of a given optimization algorithm,

    Parameters
    ----------
    dataset : Iterable[Sequence]
        Instances of the given data set.
    labels : Iterable[Labels]
        Labels for the given data set.
    spaces : list[SearchSpace]
        Search space for a given algorithm.
    metric : str, optional
        Metric to optimize, by default "f1".
    trials : int, optional
        Number of trials for each search space, by default 10.
    cv : int, optional
        Number of folds for cross-validation, by default 5.
    seed : Optional[int], optional
        Random seed, by default None.

    Returns
    -------
    dict[str, Union[str, int, float, bool]]
        [description]
    """
    # set random seed
    random.seed(seed)

    # split data set for cross validation
    splits = list(cross_validation(dataset, labels, n=cv))

    results = []
    for space in spaces:
        for trial in range(trials):
            LOGGER.info(f"Trial {trial + 1} for {space.algorithm}")

            precision_scores = []
            recall_scores = []
            f1_scores = []
            times = []

            # randomly select hyperparameters
            params = space.random_parameters()

            for (train_dataset, train_labels), (test_dataset, test_labels) in splits:
                # fire!
                start = time.time()
                trainer = Trainer(max_iterations=10, **params)
                trainer.train(train_dataset, train_labels, model_filepath="optimization.crf")
                end = time.time()

                # evaluate model
                model = Model("optimization.crf")

                # tbd
                predicted_labels = model.predict(test_dataset)

                # tbd
                scores = evaluate(test_labels, predicted_labels)

                # tbd
                precision_scores.append(scores["precision"])
                recall_scores.append(scores["recall"])
                f1_scores.append(scores["f1"])
                times.append(end - start)

            # save results
            results.append(
                {
                    "mean_precision": statistics.mean(precision_scores),
                    "stdev_precision": statistics.stdev(precision_scores),
                    "mean_recall": statistics.mean(recall_scores),
                    "stdev_recall": statistics.stdev(recall_scores),
                    "mean_f1": statistics.mean(f1_scores),
                    "stdev_f1": statistics.stdev(f1_scores),
                    "mean_time": statistics.mean(times),
                    "stdev_time": statistics.stdev(times),
                }
                | params
            )

    # sort results descending by score
    return sorted(results, key=itemgetter(f"mean_{metric}"), reverse=True)
