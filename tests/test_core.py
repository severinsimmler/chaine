from chaine import core
from chaine.crf import Model
from chaine.data import Token, TokenSequence


def test_train():
    tokens = [["Foo"], ["Bar"]]
    sequence = TokenSequence(tokens)

    dataset = [sequence]
    labels = [["O", "O"]]

    crf = core.train(dataset, labels)

    assert isinstance(crf, Model)
    assert crf.labels == {"O"}
