from chaine import core
from chaine.crf import Model


def test_train():
    sequences = [[{"foo"}, {"bar"}] for _ in range(50)]
    labels = [["O", "O"] for _ in range(50)]

    crf = core.train(sequences, labels)

    assert isinstance(crf, Model)
    assert crf.labels == {"O"}
    assert crf.predict(sequences) == labels
