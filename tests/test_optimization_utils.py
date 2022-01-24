import pytest

from chaine.optimization import utils


@pytest.fixture
def dataset():
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.fixture
def labels():
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_number_series():
    assert list(utils.NumberSeries(0, 3, 1)) == [0, 1, 2, 3]
    assert list(utils.NumberSeries(0, 1, 0.1)) == [
        0.0,
        0.1,
        0.2,
        0.30000000000000004,
        0.4,
        0.5,
        0.6000000000000001,
        0.7000000000000001,
        0.8,
        0.9,
        1.0,
    ]


def test_5_fold_cross_validation(dataset, labels):
    for train, test in utils.cross_validation(dataset, labels, k=5):
        assert len(train[0]) == 8
        assert len(train[1]) == 8
        assert len(test[0]) == 2
        assert len(test[1]) == 2


def test_2_fold_cross_validation(dataset, labels):
    for train, test in utils.cross_validation(dataset, labels, k=2):
        assert len(train[0]) == 5
        assert len(train[1]) == 5
        assert len(test[0]) == 5
        assert len(test[1]) == 5


def test_downsample(dataset, labels):
    assert len(utils.downsample(dataset, labels, n=5)[0]) == 5
    assert len(utils.downsample(dataset, labels, n=5)[1]) == 5
    assert len(utils.downsample(dataset, labels, n=2)[0]) == 2
    assert len(utils.downsample(dataset, labels, n=2)[1]) == 2

    with pytest.raises(ValueError):
        utils.downsample(dataset, labels, n=15)
