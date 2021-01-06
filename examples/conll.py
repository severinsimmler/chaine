from typing import Any, Dict, List, Union

import datasets
from seqeval.metrics import classification_report

import chaine
from chaine.logging import Logger

Sentence = List[str]
Tags = List[str]
Features = Dict[str, Union[float, int, str, bool]]
Dataset = Dict[str, Dict[str, Any]]

LOGGER = Logger(__name__)


def featurize_token(token_index: int, sentence: Sentence, pos_tags: Tags) -> Features:
    """Extract features from a token in a sentence

    Parameters
    ----------
    token_index : int
        todo
    sentence : Sentence
        todo
    pos_tags : Tags
        todo

    Returns
    -------
    Features
        todo
    """
    token = sentence[token_index]
    pos_tag = pos_tags[token_index]

    features = {
        "bias": 1.0,
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
        todo
    pos_tags : Tags
        todo

    Returns
    -------
    List[Features]
        todo
    """
    return [
        featurize_token(token_index, sentence, pos_tags)
        for token_index in range(len(sentence))
    ]


def featurize_dataset(dataset: Dataset) -> List[List[Features]]:
    """Extract features from tokens in a sentence

    Parameters
    ----------
    dataset : Dataset
        todo

    Returns
    -------
    List[List[Features]]
        todo
    """
    return [
        featurize_sentence(sentence, pos_tags)
        for sentence, pos_tags in zip(dataset["tokens"], dataset["pos_tags"])
    ]

def preprocess_labels(dataset: Dataset) -> List[List[str]]:
    labels = dataset.features["ner_tags"].feature.names
    return [[labels[index] for index in indices] for indices in dataset["ner_tags"]]


if __name__ == "__main__":
    LOGGER.info("Loading raw dataset")
    dataset = datasets.load_dataset("conll2003")

    LOGGER.info("Extracting features from dataset for training")
    sentences = featurize_dataset(dataset["train"])
    labels = preprocess_labels(dataset["train"])

    model = chaine.train(sentences, labels)

    LOGGER.info("Extracting features from dataset for evaluation")
    sentences = featurize_dataset(dataset["test"])
    labels = preprocess_labels(dataset["test"])

    LOGGER.info("Evaluating the model")
    predictions = model.predict(sentences)

    print("\nEvaluation:")
    print(classification_report(labels, predictions))
