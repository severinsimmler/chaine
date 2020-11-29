from chaine import core
from chaine.data import Sequence, Token
from chaine.model import CRF


def test_train():
    tokens = [Token(0, "Foo"), Token(1, "bar")]
    sequence = Sequence(tokens)

    dataset = [sequence]
    labels = [["O", "O"]]

    crf = core.train(dataset, labels)

    assert isinstance(crf, CRF)
    assert crf.labels == {"O"}
    assert crf.predict(sequence.featurize()) == ["O", "O"]
