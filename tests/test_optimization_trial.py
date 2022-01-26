import pytest

from chaine.optimization.spaces import L2SGDSearchSpace
from chaine.optimization.trial import OptimizationTrial
from chaine.optimization.utils import cross_validation


@pytest.fixture
def sequences():
    return [[{"a": "foo"}, {"a": "bar"}] for _ in range(50)]


@pytest.fixture
def labels():
    return [["A", "B"] for _ in range(50)]


@pytest.fixture
def splits(sequences, labels):
    return cross_validation(sequences, labels, k=5)


@pytest.fixture
def space():
    return L2SGDSearchSpace()


def test_optimization_trial(splits, space):
    with OptimizationTrial(splits, space, is_baseline=False) as trial:
        result = trial

    assert "hyperparameters" in result
    assert "stats" in result
    assert len(result["stats"]) == 8


def test_optimization_trial_baseline(splits, space):
    with OptimizationTrial(splits, space, is_baseline=True) as trial:
        result = trial

    assert "hyperparameters" in result
    assert "stats" in result
    assert len(result["stats"]) == 8
