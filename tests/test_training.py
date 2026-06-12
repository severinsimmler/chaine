import pytest

from chaine import training
from chaine.crf import Model


@pytest.fixture
def sequences():
    return [[{"word": "foo", "pos": "NN"}, {"word": "bar", "pos": "VB"}] for _ in range(50)]


@pytest.fixture
def labels():
    return [["B-PER", "I-PER"] for _ in range(50)]


@pytest.fixture
def mixed_sequences():
    """More complex sequences with varying lengths."""
    return [
        [{"word": "John", "pos": "NNP"}, {"word": "Smith", "pos": "NNP"}],
        [
            {"word": "The", "pos": "DT"},
            {"word": "quick", "pos": "JJ"},
            {"word": "brown", "pos": "JJ"},
            {"word": "fox", "pos": "NN"},
        ],
        [{"word": "Hello", "pos": "UH"}],
    ]


@pytest.fixture
def mixed_labels():
    return [
        ["B-PER", "I-PER"],
        ["O", "O", "O", "O"],
        ["O"],
    ]


def test_train(sequences, labels):
    crf = training.train(sequences, labels)

    assert isinstance(crf, Model)
    assert crf.labels == {"B-PER", "I-PER"}
    predictions = crf.predict(sequences)
    assert predictions == labels
    # ensure predictions have correct length and valid labels
    for pred_seq, orig_seq in zip(predictions, sequences, strict=True):
        assert len(pred_seq) == len(orig_seq)
        assert all(label in crf.labels for label in pred_seq)


def test_train_optimize_hyperparameters(sequences, labels):
    crf = training.train(sequences, labels, optimize_hyperparameters=True)

    assert isinstance(crf, Model)
    assert crf.labels == {"B-PER", "I-PER"}
    predictions = crf.predict(sequences)
    assert predictions == labels
    # ensure predictions have correct length and valid labels
    for pred_seq, orig_seq in zip(predictions, sequences, strict=True):
        assert len(pred_seq) == len(orig_seq)
        assert all(label in crf.labels for label in pred_seq)


@pytest.mark.parametrize("algorithm", ["lbfgs", "l2sgd", "ap", "pa", "arow"])
def test_train_all_algorithms(sequences, labels, algorithm):
    crf = training.train(sequences, labels, algorithm=algorithm)

    assert isinstance(crf, Model)
    assert crf.labels == {"B-PER", "I-PER"}
    predictions = crf.predict(sequences)
    # for training data, predictions should match labels for simple cases
    assert predictions == labels
    # ensure predictions have correct length and valid labels
    for pred_seq, orig_seq in zip(predictions, sequences, strict=True):
        assert len(pred_seq) == len(orig_seq)
        assert all(label in crf.labels for label in pred_seq)


@pytest.mark.parametrize("algorithm", ["lbfgs", "l2sgd", "ap", "pa", "arow"])
def test_train_all_algorithms_mixed_data(mixed_sequences, mixed_labels, algorithm):
    crf = training.train(mixed_sequences, mixed_labels, algorithm=algorithm)

    assert isinstance(crf, Model)
    assert crf.labels == {"B-PER", "I-PER", "O"}
    predictions = crf.predict(mixed_sequences)
    # ensure predictions have correct length and valid labels
    for pred_seq, orig_seq in zip(predictions, mixed_sequences, strict=True):
        assert len(pred_seq) == len(orig_seq)
        assert all(label in crf.labels for label in pred_seq)


def test_train_with_custom_hyperparameters(sequences, labels):
    # tTest with custom hyperparameters for lbfgs
    crf = training.train(
        sequences,
        labels,
        algorithm="lbfgs",
        c1=0.1,
        c2=0.9,
        max_iterations=10,
    )

    assert isinstance(crf, Model)
    predictions = crf.predict(sequences)
    for pred_seq, orig_seq in zip(predictions, sequences, strict=True):
        assert len(pred_seq) == len(orig_seq)
        assert all(label in crf.labels for label in pred_seq)


def test_train_with_different_label_sets():
    """Test training with different label distributions."""
    sequences = [
        [{"word": "foo"}, {"word": "bar"}],
        [{"word": "baz"}, {"word": "qux"}],
    ]
    labels = [
        ["START", "END"],
        ["START", "END"],
    ]

    crf = training.train(sequences, labels)
    assert crf.labels == {"START", "END"}
    predictions = crf.predict(sequences)
    for pred_seq, orig_seq in zip(predictions, sequences, strict=True):
        assert len(pred_seq) == len(orig_seq)
        assert all(label in crf.labels for label in pred_seq)


def test_train_single_label():
    """Test training with only one label type."""
    sequences = [[{"word": "foo"}, {"word": "bar"}] for _ in range(10)]
    labels = [["O", "O"] for _ in range(10)]

    crf = training.train(sequences, labels)
    assert crf.labels == {"O"}
    predictions = crf.predict(sequences)
    assert predictions == labels
