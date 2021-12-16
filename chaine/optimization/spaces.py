import random
from abc import ABC, abstractmethod

from chaine.optimization.utils import NumberSeries
from chaine.typing import Union


class SearchSpace(ABC):
    @property
    @abstractmethod
    def algorithm(self) -> str:
        ...

    @property
    @abstractmethod
    def max_iterations(self) -> int:
        ...

    @abstractmethod
    def random_parameters(self) -> dict[str, Union[int, float, bool, str]]:
        ...


class LBFGSSearchSpace(SearchSpace):
    def __init__(
        self,
        min_freq: Union[set[int], NumberSeries] = NumberSeries(start=0, end=100, step=5),
        all_possible_states: set[bool] = {True, False},
        all_possible_transitions: set[bool] = {True, False},
        num_memories: Union[set[int], NumberSeries] = NumberSeries(start=0, end=10, step=1),
        c1: Union[set[float], NumberSeries] = NumberSeries(start=0.0, stop=1.0, step=0.1),
        c2: Union[set[float], NumberSeries] = NumberSeries(start=0.0, stop=1.0, step=0.1),
        epsilon: Union[set[float], NumberSeries] = {1e-5},
        period: Union[set[int], NumberSeries] = {10},
        delta: Union[set[float], NumberSeries] = {1e-5},
        linesearch: set[str] = {"MoreThuente", "Backtracking", "StrongBacktracking"},
        max_linesearch: Union[set[int], NumberSeries] = {20},
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

    @property
    def max_iterations(self) -> int:
        return 100

    def random_parameters(self) -> dict[str, Union[int, float, bool, str]]:
        return {
            "algorithm": self.algorithm,
            "max_iterations": self.max_iterations,
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
    min_freq: Union[set[int], NumberSeries] = {0}
    all_possible_states: set[bool] = {True, False}
    all_possible_transitions: set[bool] = {True, False}
    c2: Union[set[float], NumberSeries] = {1.0}
    period: Union[set[int], NumberSeries] = {10}
    delta: Union[set[float], NumberSeries] = {1e-5}
    calibration_eta: Union[set[float], NumberSeries] = {0.1}
    calibration_rate: Union[set[float], NumberSeries] = {2.0}
    calibration_samples: Union[set[int], NumberSeries] = {1000}
    calibration_candidates: Union[set[int], NumberSeries] = {10}
    calibration_max_trials: Union[set[int], NumberSeries] = {20}

    @property
    def algorithm(self) -> str:
        return "l2sgd"

    @property
    def max_iterations(self) -> int:
        return 100


class APSearchSpace(SearchSpace):
    algorithms: set[str] = {"lbfgs", "l2sgd", "ap", "pa", "arow"}
    min_freq: Union[set[int], NumberSeries] = {0}
    all_possible_states: set[bool] = {True, False}
    all_possible_transitions: set[bool] = {True, False}
    epsilon: Union[set[float], NumberSeries] = {1e-5}

    @property
    def algorithm(self) -> str:
        return "ap"

    @property
    def max_iterations(self) -> int:
        return 100


class PASearchSpace(SearchSpace):
    algorithms: set[str] = {"lbfgs", "l2sgd", "ap", "pa", "arow"}
    min_freq: Union[set[int], NumberSeries] = {0}
    all_possible_states: set[bool] = {True, False}
    all_possible_transitions: set[bool] = {True, False}
    epsilon: Union[set[float], NumberSeries] = {1e-5}
    pa_type: Union[set[int], NumberSeries] = {0, 1, 2}
    c: Union[set[float], NumberSeries] = {1.0}
    errors_sensitive: set[bool] = {True, False}
    averaging: set[bool] = {True, False}

    @property
    def algorithm(self) -> str:
        return "pa"

    @property
    def max_iterations(self) -> int:
        return 100


class AROWSearchSpace(SearchSpace):
    min_freq: Union[set[int], NumberSeries] = {0}
    all_possible_states: set[bool] = {True, False}
    all_possible_transitions: set[bool] = {True, False}
    epsilon: Union[set[float], NumberSeries] = {1e-5}
    variance: Union[set[float], NumberSeries] = {1.0}
    gamma: Union[set[float], NumberSeries] = {1.0}

    @property
    def algorithm(self) -> str:
        return "arow"

    @property
    def max_iterations(self) -> int:
        return 100
