from chaine import training
from chaine.crf import Model


def test_train():
    sequences = [[{"a": "foo"}, {"a": "bar"}] for _ in range(50)]
    labels = [["O", "O"] for _ in range(50)]

    crf = training.train(sequences, labels)

    assert isinstance(crf, Model)
    assert crf.labels == {"O"}
    assert crf.predict(sequences) == labels

def test_train_optimize_hyperparameters():
    sequences = [[{"a": "foo"}, {"a": "bar"}] for _ in range(50)]
    labels = [["O", "O"] for _ in range(50)]

    crf = training.train(sequences, labels, optimize_hyperparameters=True)

    assert isinstance(crf, Model)
    assert crf.labels == {"O"}
    assert crf.predict(sequences) == labels
