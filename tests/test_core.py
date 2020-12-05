from chaine import core
from chaine.data import TokenSequence, Token
from chaine.crf import Model


def test_train():
    tokens = [["Foo"], ["Bar"]]
    sequence = TokenSequence(tokens)

    dataset = [sequence]
    labels = [["O", "O"]]

    crf = core.train(dataset, labels)

    assert isinstance(crf, Model)
    assert crf.labels == {"O"}
