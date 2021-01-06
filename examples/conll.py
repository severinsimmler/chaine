from typing import Any, Dict, List, Union

import datasets
from seqeval.metrics import classification_report

import chaine
from chaine.logging import Logger

# type hints
Sentence = List[str]
Tags = List[str]
Features = Dict[str, Union[float, int, str, bool]]
Dataset = Dict[str, Dict[str, Any]]

# consistent logging
LOGGER = Logger(__name__)


def featurize_token(token_index: int, sentence: Sentence, pos_tags: Tags) -> Features:
    """Extract features from a token in a sentence

    Parameters
    ----------
    token_index : int
        Index of the token to featurize in the sentence
    sentence : Sentence
        Sequence of tokens
    pos_tags : Tags
        Sequence of part-of-speech tags corresponding to the tokens in the sentence

    Returns
    -------
    Features
        Features representing the token
    """
    token = sentence[token_index]
    pos_tag = pos_tags[token_index]

    features = {
        "token.lower()": token.lower(),
        "token[-3:]": token[-3:],
        "token[-2:]": token[-2:],
        "token.isupper()": token.isupper(),
        "token.istitle()": token.istitle(),
        "token.isdigit()": token.isdigit(),
        "pos_tag": pos_tag,
    }
    if token_index > 0:
        previous_token = sentence[token_index - 1]
        previous_pos_tag = pos_tags[token_index - 1]
        features.update(
            {
                "-1:token.lower()": previous_token.lower(),
                "-1:token.istitle()": previous_token.istitle(),
                "-1:token.isupper()": previous_token.isupper(),
                "-1:pos_tag": previous_pos_tag,
            }
        )
    else:
        features["BOS"] = True

    if token_index < len(sentence) - 1:
        next_token = sentence[token_index + 1]
        next_pos_tag = pos_tags[token_index + 1]
        features.update(
            {
                "+1:token.lower()": next_token.lower(),
                "+1:token.istitle()": next_token.istitle(),
                "+1:token.isupper()": next_token.isupper(),
                "+1:pos_tag": next_pos_tag,
            }
        )
    else:
        features["EOS"] = True

    return features


def featurize_sentence(sentence: List[str], pos_tags: List[str]) -> List[Features]:
    """Extract features from tokens in a sentence

    Parameters
    ----------
    sentence : Sentence
        Sequence of tokens
    pos_tags : Tags
        Sequence of part-of-speech tags corresponding to the tokens in the sentence

    Returns
    -------
    List[Features]
        List of features representing tokens of a sentence
    """
    return [
        featurize_token(token_index, sentence, pos_tags)
        for token_index in range(len(sentence))
    ]


def featurize_dataset(dataset: Dataset) -> List[List[Features]]:
    """Extract features from sentences in a dataset

    Parameters
    ----------
    dataset : Dataset
        Dataset to featurize

    Returns
    -------
    List[List[Features]]
        Featurized dataset
    """
    return [
        featurize_sentence(sentence, pos_tags)
        for sentence, pos_tags in zip(dataset["tokens"], dataset["pos_tags"])
    ]


def preprocess_labels(dataset: Dataset) -> List[List[str]]:
    """Translate raw labels (i.e. integers) to the respective string labels

    Parameters
    ----------
    dataset : Dataset
        Dataset to preprocess labels

    Returns
    -------
    List[List[Features]]
        Preprocessed labels
    """
    labels = dataset.features["ner_tags"].feature.names
    return [[labels[index] for index in indices] for indices in dataset["ner_tags"]]


if __name__ == "__main__":
    LOGGER.info("Loading raw dataset")
    dataset = datasets.load_dataset("conll2003")

    LOGGER.info(f"Number of sentences for training: {len(dataset['train']['tokens'])}")
    LOGGER.info(f"Number of sentences for evaluation: {len(dataset['test']['tokens'])}")

    LOGGER.info("Extracting features from dataset for training")
    sentences = featurize_dataset(dataset["train"])
    labels = preprocess_labels(dataset["train"])

    # chaine gets the featurized sentences and the labels as input for training
    model = chaine.train(
        sentences,
        labels,
        algorithm="l2sgd",  # optimization algorithm: stochastic gradient descent
        min_freq=0,  # threshold value for minimum frequency of a feature
        all_possible_states=False,  # allow states not occuring in the data
        all_possible_transitions=False,  # allow transitions not occuring in the data
        max_iterations=100,  # number of iterations
        c2=1.0,  # coefficient for L2 regularization
        period=10,  # threshold value for iterations to test the stopping criterion
        delta=1e-5,  # top iteration when log likelihood is not greater than this
        calibration_eta=0.1,  # initial value of learning rate
        calibration_rate=2.0,  # rate of increase/decrease of learning rate
        calibration_samples=1000,  # number of instances used for calibration
        calibration_candidates=10,  # number of candidates of learning rate
        calibration_max_trials=20,  # number of trials of learning rates for calibration
    )

    LOGGER.info("Extracting features from dataset for evaluation")
    sentences = featurize_dataset(dataset["test"])
    labels = preprocess_labels(dataset["test"])

    LOGGER.info("Evaluating the trained model")
    predictions = model.predict(sentences)

    print("\n", classification_report(labels, predictions))
