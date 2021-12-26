import random
from abc import ABC, abstractmethod

from chaine.optimization.utils import NumberSeries
from chaine.typing import Union


class SearchSpace(ABC):
    @property
    @abstractmethod
    def algorithm(self) -> str:
        ...

    @abstractmethod
    def random_hyperparameters(self) -> dict[str, Union[int, float, bool, str]]:
        ...


class LBFGSSearchSpace(SearchSpace):
    def __init__(
        self,
        min_freq: NumberSeries = NumberSeries(start=0, stop=100, step=1),
        num_memories: NumberSeries = NumberSeries(start=0, stop=20, step=1),
        c1: NumberSeries = NumberSeries(start=0.0, stop=2.0, step=0.01),
        c2: NumberSeries = NumberSeries(start=0.0, stop=2.0, step=0.01),
        epsilon: NumberSeries = NumberSeries(start=0.00001, stop=0.1, step=0.00001),
        period: NumberSeries = NumberSeries(start=0, stop=20, step=1),
        delta: NumberSeries = NumberSeries(start=0.00001, stop=0.1, step=0.00001),
        max_linesearch: NumberSeries = NumberSeries(start=0, stop=50, step=1),
        linesearch: set[str] = {"MoreThuente", "Backtracking", "StrongBacktracking"},
        all_possible_states: set[bool] = {True, False},
        all_possible_transitions: set[bool] = {True, False},
    ):
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

    def random_hyperparameters(self) -> dict[str, Union[int, float, bool, str]]:
        """Select random hyperparameters from the search space.

        Returns
        -------
        dict[str, Union[int, float, bool, str]]
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
        min_freq: NumberSeries = NumberSeries(start=0, stop=100, step=1),
        all_possible_states: set[bool] = {True, False},
        all_possible_transitions: set[bool] = {True, False},
        c2: NumberSeries = NumberSeries(start=0.0, stop=2.0, step=0.01),
        period: NumberSeries = NumberSeries(start=0, stop=20, step=1),
        delta: NumberSeries = NumberSeries(start=0.00001, stop=0.1, step=0.00001),
        calibration_eta: NumberSeries = NumberSeries(start=0.00001, stop=0.1, step=0.00001),
        calibration_rate: NumberSeries = NumberSeries(start=0.0, stop=5.0, step=0.1),
        calibration_samples: NumberSeries = NumberSeries(start=100, stop=5000, step=10),
        calibration_candidates: NumberSeries = NumberSeries(start=1, stop=50, step=1),
        calibration_max_trials: NumberSeries = NumberSeries(start=1, stop=50, step=1),
    ):
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

    def random_hyperparameters(self) -> dict[str, Union[int, float, bool, str]]:
        """Select random hyperparameters from the search space.

        Returns
        -------
        dict[str, Union[int, float, bool, str]]
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
        min_freq: NumberSeries = NumberSeries(start=0, stop=100, step=1),
        all_possible_states: set[bool] = {True, False},
        all_possible_transitions: set[bool] = {True, False},
        epsilon: NumberSeries = NumberSeries(start=0.00001, stop=0.1, step=0.00001),
    ):
        self.min_freq = min_freq
        self.all_possible_states = all_possible_states
        self.all_possible_transitions = all_possible_transitions
        self.epsilon = epsilon

    @property
    def algorithm(self) -> str:
        return "ap"

    def random_hyperparameters(self) -> dict[str, Union[int, float, bool, str]]:
        """Select random hyperparameters from the search space.

        Returns
        -------
        dict[str, Union[int, float, bool, str]]
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
        min_freq: NumberSeries = NumberSeries(start=0, stop=100, step=1),
        all_possible_states: set[bool] = {True, False},
        all_possible_transitions: set[bool] = {True, False},
        epsilon: NumberSeries = NumberSeries(start=0.00001, stop=0.1, step=0.00001),
        pa_type: NumberSeries = {0, 1, 2},
        c: NumberSeries = NumberSeries(start=0.0, stop=2.0, step=0.01),
        error_sensitive: set[bool] = {True, False},
        averaging: set[bool] = {True, False},
    ):
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

    def random_hyperparameters(self) -> dict[str, Union[int, float, bool, str]]:
        """Select random hyperparameters from the search space.

        Returns
        -------
        dict[str, Union[int, float, bool, str]]
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
        min_freq: NumberSeries = NumberSeries(start=0, stop=100, step=1),
        all_possible_states: set[bool] = {True, False},
        all_possible_transitions: set[bool] = {True, False},
        epsilon: NumberSeries = NumberSeries(start=0.00001, stop=0.1, step=0.00001),
        variance: NumberSeries = NumberSeries(start=0.00001, stop=0.1, step=0.00001),
        gamma: NumberSeries = NumberSeries(start=0.00001, stop=0.1, step=0.00001),
    ):
        self.min_freq = min_freq
        self.all_possible_states = all_possible_states
        self.all_possible_transitions = all_possible_transitions
        self.epsilon = epsilon
        self.variance = variance
        self.gamma = gamma

    @property
    def algorithm(self) -> str:
        return "arow"

    def random_hyperparameters(self) -> dict[str, Union[int, float, bool, str]]:
        """Select random hyperparameters from the search space.

        Returns
        -------
        dict[str, Union[int, float, bool, str]]
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
