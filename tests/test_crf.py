import json
from pathlib import Path

import pytest

from chaine import crf


@pytest.fixture
def dataset() -> dict[str, list]:
    sequences = [[{"a": "foo"}, {"a": "bar"}] for _ in range(50)]
    labels = [["A", "B"] for _ in range(50)]
    return {"sequences": sequences, "labels": labels}


@pytest.fixture
def serialized_model(tmpdir: Path, dataset: dict[str, list]) -> Path:
    trainer = crf.Trainer()
    model_filepath = Path(tmpdir.join("model.chaine"))
    trainer.train(dataset["sequences"], dataset["labels"], model_filepath=model_filepath)
    return model_filepath


@pytest.fixture
def model(serialized_model: Path):
    return crf.Model(serialized_model)


def test_trainer_algorithm_selection():
    for algorithm in {
        "lbfgs",
        "limited-memory-bfgs",
        "l2sgd",
        "stochastic-gradient-descent",
        "ap",
        "averaged-perceptron",
        "pa",
        "passive-aggressive",
        "arow",
    }:
        trainer = crf.Trainer(algorithm)
        assert len(trainer.params) > 0

    with pytest.raises(ValueError):
        crf.Trainer("foo")


def test_special_param_values():
    trainer = crf.Trainer(
        "lbfgs",
        min_freq=1000,
        all_possible_states=True,
        all_possible_transitions=True,
        max_iterations=50,
    )

    assert trainer.params["min_freq"] == 1000
    assert trainer.params["all_possible_states"] == True
    assert trainer.params["all_possible_transitions"] == True
    assert trainer.params["max_iterations"] == 50


def test_lbfgs_params():
    trainer = crf.Trainer("lbfgs")
    for param in {
        "min_freq",
        "all_possible_states",
        "all_possible_transitions",
        "max_iterations",
        "num_memories",
        "c1",
        "c2",
        "epsilon",
        "period",
        "delta",
        "linesearch",
        "max_linesearch",
    }:
        assert param in trainer.params.keys()


def test_l2sgd_params():
    trainer = crf.Trainer("l2sgd")
    for param in {
        "min_freq",
        "all_possible_states",
        "all_possible_transitions",
        "max_iterations",
        "c2",
        "period",
        "delta",
        "calibration_eta",
        "calibration_rate",
        "calibration_samples",
        "calibration_candidates",
        "calibration_max_trials",
    }:
        assert param in trainer.params.keys()


def test_ap_params():
    trainer = crf.Trainer("ap")
    for param in {
        "min_freq",
        "all_possible_states",
        "all_possible_transitions",
        "max_iterations",
        "epsilon",
    }:
        assert param in trainer.params.keys()


def test_pa_params():
    trainer = crf.Trainer("pa")
    for param in {
        "min_freq",
        "all_possible_states",
        "all_possible_transitions",
        "max_iterations",
        "epsilon",
        "pa_type",
        "c",
        "error_sensitive",
        "averaging",
    }:
        assert param in trainer.params.keys()


def test_arow_params():
    trainer = crf.Trainer("arow")
    for param in {
        "min_freq",
        "all_possible_states",
        "all_possible_transitions",
        "max_iterations",
        "epsilon",
        "variance",
        "gamma",
    }:
        assert param in trainer.params.keys()


def test_training(tmpdir, dataset: dict[str, list]):
    trainer = crf.Trainer()
    model_filepath = Path(tmpdir.join("model.chaine"))
    assert not model_filepath.exists()

    trainer.train(dataset["sequences"], dataset["labels"], model_filepath=model_filepath)
    assert model_filepath.exists()


def test_model_deserialization(serialized_model):
    model = crf.Model(serialized_model)
    assert model.labels == {"A", "B"}


def test_model_predict_single(model: crf.Model, dataset: dict[str, list]):
    for sequence in dataset["sequences"]:
        predicted = model.predict_single(sequence)
        expected = ["A", "B"]
        assert predicted == expected

    with pytest.raises(ValueError):
        model.predict_single(dataset["sequences"])


def test_model_predict(model: crf.Model, dataset: dict[str, list]):
    predicted = model.predict(dataset["sequences"])
    expected = dataset["labels"]
    assert predicted == expected


def test_model_predict_generator(model: crf.Model, dataset: dict[str, list]):
    generator = ([features for features in sequence] for sequence in dataset["sequences"])
    predicted = model.predict(generator)
    expected = dataset["labels"]
    assert predicted == expected


def test_model_predict_proba_single(model: crf.Model, dataset: dict[str, list]):
    for sequence in dataset["sequences"]:
        predicted = model.predict_proba_single(sequence)
        expected = [
            {"A": 0.953079109284954, "B": 0.04692089071504606},
            {"A": 0.046920890715046036, "B": 0.953079109284954},
        ]
        assert predicted == expected

    with pytest.raises(ValueError):
        model.predict_proba_single(dataset["sequences"])


def test_model_predict_proba(model: crf.Model, dataset: dict[str, list]):
    predicted = model.predict_proba(dataset["sequences"])
    expected = [
        [
            {"A": 0.953079109284954, "B": 0.04692089071504606},
            {"A": 0.046920890715046036, "B": 0.953079109284954},
        ]
        for _ in dataset["labels"]
    ]
    assert predicted == expected


def test_model_predict_proba_generator(model: crf.Model, dataset: dict[str, list]):
    generator = ([features for features in sequence] for sequence in dataset["sequences"])
    predicted = model.predict_proba(generator)
    expected = [
        [
            {"A": 0.953079109284954, "B": 0.04692089071504606},
            {"A": 0.046920890715046036, "B": 0.953079109284954},
        ]
        for _ in dataset["labels"]
    ]
    assert predicted == expected


def test_dump_transitions(model: crf.Model):
    filepath = Path("transitions.json")

    model.dump_transitions(filepath)
    assert filepath.exists()

    transitions = json.loads(filepath.read_text())
    assert len(transitions) > 0


def test_dump_states(model: crf.Model):
    filepath = Path("states.json")

    model.dump_states(filepath)
    assert filepath.exists()

    states = json.loads(filepath.read_text())
    assert len(states) > 0


def test_optimizer(dataset: dict[str, list]):
    optimizer = crf.Optimizer()
    result = optimizer.optimize_hyperparameters(dataset["sequences"], dataset["labels"])

    assert len(result) > 0
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert len(result[0]) == 2
    assert "hyperparameters" in result[0]
    assert "stats" in result[0]
