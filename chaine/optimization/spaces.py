"""
chaine.optimization.spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements hyperparameter search spaces for the different training methods.
"""

import random
from abc import ABC, abstractmethod

from chaine.optimization.utils import NumberSeries


class SearchSpace(ABC):
    @property
    @abstractmethod
    def algorithm(self) -> str:
        ...

    @abstractmethod
    def random_hyperparameters(self) -> dict[str, int | float | bool | str]:
        ...


class LBFGSSearchSpace(SearchSpace):
    def __init__(
        self,
        min_freq: NumberSeries = NumberSeries(start=0, stop=5, step=1),
        num_memories: NumberSeries = NumberSeries(start=1, stop=10, step=1),
        c1: NumberSeries = NumberSeries(start=0.0, stop=2.0, step=0.01),
        c2: NumberSeries = NumberSeries(start=0.0, stop=2.0, step=0.01),
        epsilon: NumberSeries = NumberSeries(start=0.00001, stop=0.001, step=0.00001),
        period: NumberSeries = NumberSeries(start=1, stop=20, step=1),
        delta: NumberSeries = NumberSeries(start=0.00001, stop=0.001, step=0.00001),
        max_linesearch: NumberSeries = NumberSeries(start=0, stop=50, step=1),
        linesearch: set[str] = {"MoreThuente", "Backtracking", "StrongBacktracking"},
        all_possible_states: set[bool] = {True, False},
        all_possible_transitions: set[bool] = {True, False},
    ):
        """Hyperparameter search space for Limited-Memory BFGS.

        Parameters
        ----------
        min_freq : NumberSeries, optional
            Threshold value for minimum frequency of a feature occurring in training data,
            by default NumberSeries(start=0, stop=5, step=1).
        num_memories : NumberSeries, optional
            Number of limited memories for approximating the inverse hessian matrix,
            by default NumberSeries(start=1, stop=10, step=1)
        c1 : NumberSeries, optional
            Coefficient for L1 regularization,
            by default NumberSeries(start=0.0, stop=2.0, step=0.01).
        c2 : NumberSeries, optional
            Coefficient for L2 regularization,
            by default NumberSeries(start=0.0, stop=2.0, step=0.01).
        epsilon : NumberSeries, optional
            Parameter that determines the condition of convergence,
            by default NumberSeries(start=0.00001, stop=0.001, step=0.00001).
        period : NumberSeries, optional
            Threshold value for iterations to test the stopping criterion,
            by default NumberSeries(start=1, stop=20, step=1).
        delta : NumberSeries, optional
            Top iteration when log likelihood is not greater than this,
            by default NumberSeries(start=0.00001, stop=0.001, step=0.00001).
        max_linesearch : NumberSeries, optional
            Maximum number of trials for the line search algorithm,
            by default NumberSeries(start=0, stop=50, step=1).
        linesearch : set[str], optional
            Line search algorithm used in updates,
            by default {"MoreThuente", "Backtracking", "StrongBacktracking"}.
        all_possible_states : set[bool], optional
            Generate state features that do not even occur in the training data,
            by default {True, False}.
        all_possible_transitions : set[bool], optional
            Generate transition features that do not even occur in the training data,
            by default {True, False}.
        """
        self.min_freq = min_freq
        self.all_possible_states = all_possible_states
        self.all_possible_transitions = all_possible_transitions
        self.num_memories = num_memories
        self.c1 = c1
        self.c2 = c2
        self.epsilon = epsilon
        self.period = period
        self.delta = delta
        self.linesearch = linesearch
        self.max_linesearch = max_linesearch

    @property
    def algorithm(self) -> str:
        return "lbfgs"

    def random_hyperparameters(self) -> dict[str, int | float | bool | str]:
        """Select random hyperparameters from the search space.

        Returns
        -------
        dict[str, int | float | bool | str]
            Randomly selected hyperparameters.
        """
        return {
            "algorithm": self.algorithm,
            "min_freq": random.choice(list(self.min_freq)),
            "all_possible_states": random.choice(list(self.all_possible_states)),
            "all_possible_transitions": random.choice(list(self.all_possible_transitions)),
            "num_memories": random.choice(list(self.num_memories)),
            "c1": random.choice(list(self.c1)),
            "c2": random.choice(list(self.c2)),
            "epsilon": random.choice(list(self.epsilon)),
            "period": random.choice(list(self.period)),
            "delta": random.choice(list(self.delta)),
            "linesearch": random.choice(list(self.linesearch)),
            "max_linesearch": random.choice(list(self.max_linesearch)),
        }


class L2SGDSearchSpace(SearchSpace):
    def __init__(
        self,
        min_freq: NumberSeries = NumberSeries(start=0, stop=5, step=1),
        all_possible_states: set[bool] = {True, False},
        all_possible_transitions: set[bool] = {True, False},
        c2: NumberSeries = NumberSeries(start=0.0, stop=2.0, step=0.01),
        period: NumberSeries = NumberSeries(start=1, stop=20, step=1),
        delta: NumberSeries = NumberSeries(start=0.00001, stop=0.001, step=0.00001),
        calibration_eta: NumberSeries = NumberSeries(start=0.00001, stop=0.001, step=0.00001),
        calibration_rate: NumberSeries = NumberSeries(start=0.5, stop=5.0, step=0.1),
        calibration_samples: NumberSeries = NumberSeries(start=100, stop=3000, step=10),
        calibration_candidates: NumberSeries = NumberSeries(start=1, stop=30, step=1),
        calibration_max_trials: NumberSeries = NumberSeries(start=1, stop=30, step=1),
    ):
        """Hyperparameter search space for SGD with L2 parameters.

        Parameters
        ----------
        min_freq : NumberSeries, optional
            Threshold value for minimum frequency of a feature occurring in training data,
            by default NumberSeries(start=0, stop=5, step=1).
        all_possible_states : set[bool], optional
            Generate state features that do not even occur in the training data,
            by default {True, False}.
        all_possible_transitions : set[bool], optional
            Generate transition features that do not even occur in the training data,
            by default {True, False}.
        c2 : NumberSeries, optional
            Coefficient for L2 regularization,
            by default NumberSeries(start=0.0, stop=2.0, step=0.01).
        period : NumberSeries, optional
            Threshold value for iterations to test the stopping criterion,
            by default NumberSeries(start=1, stop=20, step=1).
        delta : NumberSeries, optional
            Top iteration when log likelihood is not greater than this,
            by default NumberSeries(start=0.00001, stop=0.001, step=0.00001).
        calibration_eta : NumberSeries, optional
            Initial value of learning rate (eta) used for calibration,
            by default NumberSeries(start=0.00001, stop=0.001, step=0.00001).
        calibration_rate : NumberSeries, optional
            Rate of increase/decrease of learning rate for calibration,
            by default NumberSeries(start=0.5, stop=5.0, step=0.1).
        calibration_samples : NumberSeries, optional
            Number of instances used for calibration,
            by default NumberSeries(start=100, stop=3000, step=10).
        calibration_candidates : NumberSeries, optional
            Number of candidates of learning rate,
            by default NumberSeries(start=1, stop=30, step=1).
        calibration_max_trials : NumberSeries, optional
            Maximum number of trials of learning rates for calibration,
            by default NumberSeries(start=1, stop=30, step=1).
        """
        self.min_freq = min_freq
        self.all_possible_states = all_possible_states
        self.all_possible_transitions = all_possible_transitions
        self.c2 = c2
        self.period = period
        self.delta = delta
        self.calibration_eta = calibration_eta
        self.calibration_rate = calibration_rate
        self.calibration_samples = calibration_samples
        self.calibration_candidates = calibration_candidates
        self.calibration_max_trials = calibration_max_trials

    @property
    def algorithm(self) -> str:
        return "l2sgd"

    def random_hyperparameters(self) -> dict[str, int | float | bool | str]:
        """Select random hyperparameters from the search space.

        Returns
        -------
        dict[str, int | float | bool | str]
            Randomly selected hyperparameters.
        """
        return {
            "algorithm": self.algorithm,
            "min_freq": random.choice(list(self.min_freq)),
            "all_possible_states": random.choice(list(self.all_possible_states)),
            "all_possible_transitions": random.choice(list(self.all_possible_transitions)),
            "c2": random.choice(list(self.c2)),
            "period": random.choice(list(self.period)),
            "delta": random.choice(list(self.delta)),
            "calibration_eta": random.choice(list(self.calibration_eta)),
            "calibration_rate": random.choice(list(self.calibration_rate)),
            "calibration_samples": random.choice(list(self.calibration_samples)),
            "calibration_candidates": random.choice(list(self.calibration_candidates)),
            "calibration_max_trials": random.choice(list(self.calibration_max_trials)),
        }


class APSearchSpace(SearchSpace):
    def __init__(
        self,
        min_freq: NumberSeries = NumberSeries(start=0, stop=5, step=1),
        all_possible_states: set[bool] = {True, False},
        all_possible_transitions: set[bool] = {True, False},
        epsilon: NumberSeries = NumberSeries(start=0.00001, stop=0.001, step=0.00001),
    ):
        """Hyperparameter search space for Averaged Perceptron.

        Parameters
        ----------
        min_freq : NumberSeries, optional
            Threshold value for minimum frequency of a feature occurring in training data,
            by default NumberSeries(start=0, stop=5, step=1).
        all_possible_states : set[bool], optional
            Generate state features that do not even occur in the training data,
            by default {True, False}.
        all_possible_transitions : set[bool], optional
            Generate transition features that do not even occur in the training data,
            by default {True, False}.
        epsilon : NumberSeries, optional
            Parameter that determines the condition of convergence,
            by default NumberSeries(start=0.00001, stop=0.001, step=0.00001).
        """
        self.min_freq = min_freq
        self.all_possible_states = all_possible_states
        self.all_possible_transitions = all_possible_transitions
        self.epsilon = epsilon

    @property
    def algorithm(self) -> str:
        return "ap"

    def random_hyperparameters(self) -> dict[str, int | float | bool | str]:
        """Select random hyperparameters from the search space.

        Returns
        -------
        dict[str, int | float | bool | str]
            Randomly selected hyperparameters.
        """
        return {
            "algorithm": self.algorithm,
            "min_freq": random.choice(list(self.min_freq)),
            "all_possible_states": random.choice(list(self.all_possible_states)),
            "all_possible_transitions": random.choice(list(self.all_possible_transitions)),
            "epsilon": random.choice(list(self.epsilon)),
        }


class PASearchSpace(SearchSpace):
    def __init__(
        self,
        min_freq: NumberSeries = NumberSeries(start=0, stop=5, step=1),
        all_possible_states: set[bool] = {True, False},
        all_possible_transitions: set[bool] = {True, False},
        epsilon: NumberSeries = NumberSeries(start=0.00001, stop=0.001, step=0.00001),
        pa_type: NumberSeries = {0, 1, 2},
        c: NumberSeries = NumberSeries(start=0.0, stop=2.0, step=0.01),
        error_sensitive: set[bool] = {True, False},
        averaging: set[bool] = {True, False},
    ):
        """Hyperparameter search space for Passive Aggressive.

        Parameters
        ----------
        min_freq : NumberSeries, optional
            Threshold value for minimum frequency of a feature occurring in training data,
            by default NumberSeries(start=0, stop=5, step=1).
        all_possible_states : set[bool], optional
            Generate state features that do not even occur in the training data,
            by default {True, False}.
        all_possible_transitions : set[bool], optional
            Generate transition features that do not even occur in the training data,
            by default {True, False}.
        epsilon : NumberSeries, optional
            Parameter that determines the condition of convergence,
            by default NumberSeries(start=0.00001, stop=0.001, step=0.00001).
        pa_type : NumberSeries, optional
            Strategy for updating feature weights, by default {0, 1, 2}.
        c : NumberSeries, optional
            Aggressiveness parameter, by default NumberSeries(start=0.0, stop=2.0, step=0.01).
        error_sensitive : set[bool], optional
            Include square root of predicted incorrect labels into optimization routine,
            by default {True, False}.
        averaging : set[bool], optional
            Compute average of feature weights at all updates, by default {True, False}.
        """
        self.min_freq = min_freq
        self.all_possible_states = all_possible_states
        self.all_possible_transitions = all_possible_transitions
        self.epsilon = epsilon
        self.pa_type = pa_type
        self.c = c
        self.error_sensitive = error_sensitive
        self.averaging = averaging

    @property
    def algorithm(self) -> str:
        return "pa"

    def random_hyperparameters(self) -> dict[str, int | float | bool | str]:
        """Select random hyperparameters from the search space.

        Returns
        -------
        dict[str, int | float | bool | str]
            Randomly selected hyperparameters.
        """
        return {
            "algorithm": self.algorithm,
            "min_freq": random.choice(list(self.min_freq)),
            "all_possible_states": random.choice(list(self.all_possible_states)),
            "all_possible_transitions": random.choice(list(self.all_possible_transitions)),
            "epsilon": random.choice(list(self.epsilon)),
            "pa_type": random.choice(list(self.pa_type)),
            "c": random.choice(list(self.c)),
            "error_sensitive": random.choice(list(self.error_sensitive)),
            "averaging": random.choice(list(self.averaging)),
        }


class AROWSearchSpace(SearchSpace):
    def __init__(
        self,
        min_freq: NumberSeries = NumberSeries(start=0, stop=5, step=1),
        all_possible_states: set[bool] = {True, False},
        all_possible_transitions: set[bool] = {True, False},
        epsilon: NumberSeries = NumberSeries(start=0.00001, stop=0.001, step=0.00001),
        variance: NumberSeries = NumberSeries(start=0.00001, stop=0.001, step=0.00001),
        gamma: NumberSeries = NumberSeries(start=0.00001, stop=0.001, step=0.00001),
    ):
        """Hyperparameter search space for AROW.

        Parameters
        ----------
        min_freq : NumberSeries, optional
            Threshold value for minimum frequency of a feature occurring in training data,
            by default NumberSeries(start=0, stop=5, step=1).
        all_possible_states : set[bool], optional
            Generate state features that do not even occur in the training data,
            by default {True, False}.
        all_possible_transitions : set[bool], optional
            Generate transition features that do not even occur in the training data,
            by default {True, False}.
        epsilon : NumberSeries, optional
            Parameter that determines the condition of convergence,
            by default NumberSeries(start=0.00001, stop=0.001, step=0.00001).
        variance : NumberSeries, optional
            Initial variance of every feature weight,
            by default NumberSeries(start=0.00001, stop=0.001, step=0.00001).
        gamma : NumberSeries, optional
            Trade-off between loss function and changes of feature weights,
            by default NumberSeries(start=0.00001, stop=0.001, step=0.00001).
        """
        self.min_freq = min_freq
        self.all_possible_states = all_possible_states
        self.all_possible_transitions = all_possible_transitions
        self.epsilon = epsilon
        self.variance = variance
        self.gamma = gamma

    @property
    def algorithm(self) -> str:
        return "arow"

    def random_hyperparameters(self) -> dict[str, int | float | bool | str]:
        """Select random hyperparameters from the search space.

        Returns
        -------
        dict[str, int | float | bool | str]
            Randomly selected hyperparameters.
        """
        return {
            "algorithm": self.algorithm,
            "min_freq": random.choice(list(self.min_freq)),
            "all_possible_states": random.choice(list(self.all_possible_states)),
            "all_possible_transitions": random.choice(list(self.all_possible_transitions)),
            "epsilon": random.choice(list(self.epsilon)),
            "variance": random.choice(list(self.variance)),
            "gamma": random.choice(list(self.gamma)),
        }
