from pathlib import Path
import pytest
from chaine import crf


@pytest.fixture
def dataset():
    sequences = [[{"foo"}, {"bar"}] for _ in range(50)]
    labels = [["O", "O"] for _ in range(50)]
    return {"sequences": sequences, "labels": labels}


@pytest.fixture
def serialized_model(tmpdir, dataset):
    trainer = crf.Trainer()
    model_filepath = Path(tmpdir.join("model.crf"))
    trainer.train(dataset["sequences"], dataset["labels"], model_filepath)
    return model_filepath


@pytest.fixture
def model(serialized_model):
    return crf.Model(serialized_model)


def test_intbool():
    value = crf._intbool("0")
    assert isinstance(value, bool)
    assert value == False

    value = crf._intbool("1")
    assert isinstance(value, bool)
    assert value == True

    with pytest.raises(ValueError):
        crf._intbool("foo")


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

    with pytest.raises(KeyError):
        crf.Trainer("foo")


def test_data_append(dataset):
    trainer = crf.Trainer()
    for sequence, label in zip(dataset["sequences"], dataset["labels"]):
        trainer._append(sequence, label)


def test_data_generator_append(dataset):
    sequences = ((token for token in sequence) for sequence in dataset["sequences"])
    labels = ((label for label in labels) for labels in dataset["labels"])

    trainer = crf.Trainer()
    for sequence, label in zip(sequences, labels):
        trainer._append(sequence, label)


def test_integer_label_append(dataset):
    labels = [[0 for _ in labels] for labels in dataset["labels"]]

    trainer = crf.Trainer()
    for sequence, label in zip(dataset["sequences"], labels):
        trainer._append(sequence, label)


def test_wrong_dataset_format(dataset):
    trainer = crf.Trainer()
    with pytest.raises(TypeError):
        trainer._append(dataset["sequences"], dataset["labels"])


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


def test_pa_params():
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


def test_log_parser(capfd):
    trainer = crf.Trainer()
    assert hasattr(trainer, "_log_parser")


def test_training(tmpdir, dataset):
    trainer = crf.Trainer()
    model_filepath = Path(tmpdir.join("model.crf"))
    assert not model_filepath.exists()

    trainer.train(dataset["sequences"], dataset["labels"], model_filepath)
    assert model_filepath.exists()


def test_model_deserialization(serialized_model):
    model = crf.Model(serialized_model)

    assert model.labels == {"O"}


def test_model_predict_single(model, dataset):
    for sequence in dataset["sequences"]:
        predicted = model.predict_single(sequence)
        expected = ["O", "O"]

        assert predicted == expected

    with pytest.raises(SystemError):
        model.predict_single(dataset["sequences"])


def test_model_predict(model, dataset):
    predicted = model.predict(dataset["sequences"])
    expected = dataset["labels"]

    assert predicted == expected


def test_model_predict_proba_single(model, dataset):
    for sequence in dataset["sequences"]:
        predicted = model.predict_proba_single(sequence)
        expected = [{"O": 1.0}, {"O": 1.0}]

        assert predicted == expected

    with pytest.raises(TypeError):
        model.predict_proba_single(dataset["sequences"])


def test_model_predict_proba(model, dataset):
    predicted = model.predict_proba(dataset["sequences"])
    expected = [[{"O": 1.0}, {"O": 1.0}] for _ in dataset["labels"]]

    assert predicted == expected
