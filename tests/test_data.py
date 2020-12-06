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
