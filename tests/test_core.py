from chaine import core
from chaine.data import TokenSequence, Token
from chaine.crf import Model


def test_train():
    tokens = [Token(0, "Foo"), Token(1, "bar")]
    sequence = TokenSequence(tokens)

    dataset = [sequence]
    labels = [["O", "O"]]

    crf = core.train(dataset, labels)

    assert isinstance(crf, Model)
    assert crf.labels == {"O"}
    assert crf.predict(sequence.featurize()) == ["O", "O"]
