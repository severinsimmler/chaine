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


def test_sequence():
    tokens = [data.Token(0, "Foo"), data.Token(1, "Bar")]
    sequence = data.Sequence(tokens)

    assert sequence.tokens == tokens
    assert sequence[0] == tokens[0]
    assert sequence[1] == tokens[1]

    with pytest.raises(IndexError):
        sequence[2]

    assert list(sequence) == tokens
    assert len(sequence) == 2
    assert repr(sequence) == "<Sequence: [<Token 0: Foo>, <Token 1: Bar>]>"
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
