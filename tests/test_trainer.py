from pathlib import Path

import pytest

from chaine.model import Trainer


@pytest.fixture
def dataset():
    sequences = [[{"foo"}, {"bar"}] for _ in range(50)]
    labels = [["O", "O"] for _ in range(50)]
    return {"sequences": sequences, "labels": labels}


def test_trainer(tmpdir, dataset):
    trainer = Trainer("lbfgs")

    model_filepath = Path(tmpdir.join("model.crf"))
    assert not model_filepath.exists()
    trainer.train(dataset["sequences"], dataset["labels"], model_filepath)
    assert model_filepath.exists()


@pytest.mark.parametrize(
    "algorithm",
    [
        "lbfgs",
        "l2sgd",
        "ap",
        "averaged-perceptron",
        "pa",
        "passive-aggressive",
        "arow",
    ],
)
def test_algorithm_parameters(algorithm):
    trainer = Trainer(algorithm)
    assert len(trainer.params) > 0


def test_params():
    trainer = Trainer("lbfgs")
    assert "c1" in trainer.params
    assert "c2" in trainer.params
    assert "num_memories" in trainer.params

    trainer = Trainer("l2sgd")
    assert "c2" in trainer.params
    assert "c1" not in trainer.params
