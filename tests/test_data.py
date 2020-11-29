import pytest

from chaine import data


def test_token():
    token = data.Token(0, "Foo")

    assert len(token) == 3
    assert repr(token) == "<Token 0: Foo>"
    assert str(token) == "Foo"
    assert token.lower() == "foo"
    assert token.is_digit == False
    assert token.is_lower == False
    assert token.is_title == True
    assert token.is_upper == False


def test_token_sequence():
    tokens = [data.Token(0, "Foo"), data.Token(1, "Bar")]
    sequence = data.TokenSequence(tokens)

    assert sequence.items == tokens
    assert sequence[0] == tokens[0]
    assert sequence[1] == tokens[1]

    with pytest.raises(IndexError):
        sequence[2]

    assert list(sequence) == tokens
    assert len(sequence) == 2
    assert repr(sequence) == "<TokenSequence: [<Token 0: Foo>, <Token 1: Bar>]>"
    assert str(sequence) == "Foo Bar"
    assert sequence.indices == [0, 1]

    assert list(sequence.featurize()) == [
        {
            "token.is_title()=True",
            "+1:token.is_upper()=False",
            "token.is_digit()=False",
            "token.is_upper()=False",
            "token.lower()=foo",
            "+1:token.lower()=bar",
            "+1:token.is_title()=True",
            "BOS=True",
        },
        {
            "EOS=True",
            "token.is_title()=True",
            "token.is_digit()=False",
            "token.is_upper()=False",
            "-1:token.is_upper()=False",
            "-1:token.is_title()=True",
            "-1:token.lower()=foo",
            "token.lower()=bar",
        },
    ]


def test_label_sequence():
    labels = ["B-PER", "I-PER", "O"]
    sequence = data.LabelSequence(labels)

    assert sequence.items == labels
    assert sequence[0] == labels[0]
    assert sequence[1] == labels[1]
    assert sequence[2] == labels[2]

    with pytest.raises(IndexError):
        sequence[3]

    assert list(sequence) == labels
    assert len(sequence) == 3
    assert repr(sequence) == "<LabelSequence: ['B-PER', 'I-PER', 'O']>"
    assert str(sequence) == "B-PER, I-PER, O"
    assert sequence == data.LabelSequence(labels)
    assert sequence != data.LabelSequence(["B-ORG", "I-ORG", "O"])
    assert sequence.distinct == set(labels)
