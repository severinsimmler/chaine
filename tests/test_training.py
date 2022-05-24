import pytest

from chaine import training
from chaine.crf import Model


@pytest.fixture
def sequences():
    return [[{"a": "foo"}, {"a": "bar"}] for _ in range(50)]


@pytest.fixture
def labels():
    return [["O", "O"] for _ in range(50)]


def test_train(sequences, labels):
    crf = training.train(sequences, labels)

    assert isinstance(crf, Model)
    assert crf.labels == {"O"}
    assert crf.predict(sequences) == labels


def test_train_optimize_hyperparameters(sequences, labels):
    crf = training.train(sequences, labels, optimize_hyperparameters=True)

    assert isinstance(crf, Model)
    assert crf.labels == {"O"}
    assert crf.predict(sequences) == labels
