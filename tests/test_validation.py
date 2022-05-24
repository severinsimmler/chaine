from chaine import validation


def test_is_valid_sequence():
    assert validation.is_valid_sequence([{"int": 0, "str": "a", "float": 1.0, "bool": True}])

    assert not validation.is_valid_sequence([[{"int": 0, "str": "a", "float": 1.0, "bool": True}]])
    assert not validation.is_valid_sequence([{"int": 0, "str": ["a"], "float": 1.0}])
    assert not validation.is_valid_sequence(["foo", "bar"])
    assert not validation.is_valid_sequence([["foo", "bar"]])
    assert not validation.is_valid_sequence({"int": 0, "str": "a", "float": 1.0, "bool": True})
    assert not validation.is_valid_sequence(({"int": 0} for _ in range(5)))


def test_is_valid_token():
    assert validation.is_valid_token({0: 0, "int": 0, "str": "a", "float": 1.0, "bool": True})

    assert not validation.is_valid_token({"int": [0], "str": "a", "float": 1.0, "bool": True})
    assert not validation.is_valid_token({"int": 0, "str": "a", "float": 1.0, "dict": {}})
    assert not validation.is_valid_token("foo")
