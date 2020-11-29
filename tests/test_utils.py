from chaine import utils


def test_log_message():
    message = utils.LogMessage()

    assert message.iteration is None
    assert message.loss is None

    message.iteration = "1"
    message.loss = "1000.0"
    assert message.iteration == "1"
    assert message.loss == "1000.0"
    assert str(message) == "Iteration: 1\tLoss: 1000.0"


def test_log_parser():
    parser = utils.LogParser()

    assert isinstance(parser.message, utils.LogMessage)

    parser.parse("Irrelevant message")
    assert parser.message.iteration is None
    assert parser.message.loss is None

    parser.parse("***** Iteration #1 *****\n")
    assert parser.message.iteration == "1"
    assert parser.message.loss is None

    text = parser.parse("Loss: 1000.0")
    assert text == "Iteration: 1\tLoss: 1000.0"
    assert parser.message.iteration is None
    assert parser.message.loss is None
