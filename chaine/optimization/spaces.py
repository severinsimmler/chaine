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
    def algorithm(self) -> str: ...

    def random_hyperparameters(self) -> dict[str, int | float | bool | str]:
        """Select random hyperparameters from the search space.

        Returns
        -------
        dict[str, int | float | bool | str]
            Randomly selected hyperparameters.
        """
        return {
            "algorithm": self.algorithm,
            **{name: random.choice(list(values)) for name, values in vars(self).items()},
        }


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
        linesearch: tuple[str, ...] = ("MoreThuente", "Backtracking", "StrongBacktracking"),
        all_possible_states: tuple[bool, ...] = (True, False),
        all_possible_transitions: tuple[bool, ...] = (True, False),
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
        linesearch : tuple[str, ...], optional
            Line search algorithm used in updates,
            by default ("MoreThuente", "Backtracking", "StrongBacktracking").
        all_possible_states : tuple[bool, ...], optional
            Generate state features that do not even occur in the training data,
            by default (True, False).
        all_possible_transitions : tuple[bool, ...], optional
            Generate transition features that do not even occur in the training data,
            by default (True, False).
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


class L2SGDSearchSpace(SearchSpace):
    def __init__(
        self,
        min_freq: NumberSeries = NumberSeries(start=0, stop=5, step=1),
        all_possible_states: tuple[bool, ...] = (True, False),
        all_possible_transitions: tuple[bool, ...] = (True, False),
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
        all_possible_states : tuple[bool, ...], optional
            Generate state features that do not even occur in the training data,
            by default (True, False).
        all_possible_transitions : tuple[bool, ...], optional
            Generate transition features that do not even occur in the training data,
            by default (True, False).
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


class APSearchSpace(SearchSpace):
    def __init__(
        self,
        min_freq: NumberSeries = NumberSeries(start=0, stop=5, step=1),
        all_possible_states: tuple[bool, ...] = (True, False),
        all_possible_transitions: tuple[bool, ...] = (True, False),
        epsilon: NumberSeries = NumberSeries(start=0.00001, stop=0.001, step=0.00001),
    ):
        """Hyperparameter search space for Averaged Perceptron.

        Parameters
        ----------
        min_freq : NumberSeries, optional
            Threshold value for minimum frequency of a feature occurring in training data,
            by default NumberSeries(start=0, stop=5, step=1).
        all_possible_states : tuple[bool, ...], optional
            Generate state features that do not even occur in the training data,
            by default (True, False).
        all_possible_transitions : tuple[bool, ...], optional
            Generate transition features that do not even occur in the training data,
            by default (True, False).
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


class PASearchSpace(SearchSpace):
    def __init__(
        self,
        min_freq: NumberSeries = NumberSeries(start=0, stop=5, step=1),
        all_possible_states: tuple[bool, ...] = (True, False),
        all_possible_transitions: tuple[bool, ...] = (True, False),
        epsilon: NumberSeries = NumberSeries(start=0.00001, stop=0.001, step=0.00001),
        pa_type: tuple[int, ...] = (0, 1, 2),
        c: NumberSeries = NumberSeries(start=0.0, stop=2.0, step=0.01),
        error_sensitive: tuple[bool, ...] = (True, False),
        averaging: tuple[bool, ...] = (True, False),
    ):
        """Hyperparameter search space for Passive Aggressive.

        Parameters
        ----------
        min_freq : NumberSeries, optional
            Threshold value for minimum frequency of a feature occurring in training data,
            by default NumberSeries(start=0, stop=5, step=1).
        all_possible_states : tuple[bool, ...], optional
            Generate state features that do not even occur in the training data,
            by default (True, False).
        all_possible_transitions : tuple[bool, ...], optional
            Generate transition features that do not even occur in the training data,
            by default (True, False).
        epsilon : NumberSeries, optional
            Parameter that determines the condition of convergence,
            by default NumberSeries(start=0.00001, stop=0.001, step=0.00001).
        pa_type : tuple[int, ...], optional
            Strategy for updating feature weights, by default (0, 1, 2).
        c : NumberSeries, optional
            Aggressiveness parameter, by default NumberSeries(start=0.0, stop=2.0, step=0.01).
        error_sensitive : tuple[bool, ...], optional
            Include square root of predicted incorrect labels into optimization routine,
            by default (True, False).
        averaging : tuple[bool, ...], optional
            Compute average of feature weights at all updates, by default (True, False).
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


class AROWSearchSpace(SearchSpace):
    def __init__(
        self,
        min_freq: NumberSeries = NumberSeries(start=0, stop=5, step=1),
        all_possible_states: tuple[bool, ...] = (True, False),
        all_possible_transitions: tuple[bool, ...] = (True, False),
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
        all_possible_states : tuple[bool, ...], optional
            Generate state features that do not even occur in the training data,
            by default (True, False).
        all_possible_transitions : tuple[bool, ...], optional
            Generate transition features that do not even occur in the training data,
            by default (True, False).
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
