from chaine.typing import Dataset, Features, Sentence, Tags


def featurize_token(token_index: int, sentence: Sentence, pos_tags: Tags) -> Features:
    """Extract features from a token in a sentence.

    Parameters
    ----------
    token_index : int
        Index of the token to featurize in the sentence.
    sentence : Sentence
        Sequence of tokens.
    pos_tags : Tags
        Sequence of part-of-speech tags corresponding to the tokens in the sentence.

    Returns
    -------
    Features
        Features representing the token.
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


def featurize_sentence(sentence: Sentence, pos_tags: Tags) -> list[Features]:
    """Extract features from tokens in a sentence.

    Parameters
    ----------
    sentence : Sentence
        Sequence of tokens.
    pos_tags : Tags
        Sequence of part-of-speech tags corresponding to the tokens in the sentence.

    Returns
    -------
    list[Features]
        List of features representing tokens of a sentence.
    """
    return [
        featurize_token(token_index, sentence, pos_tags) for token_index in range(len(sentence))
    ]


def featurize_dataset(dataset: Dataset) -> list[list[Features]]:
    """Extract features from sentences in a dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset to featurize.

    Returns
    -------
    list[list[Features]]
        Featurized dataset.
    """
    return [
        featurize_sentence(sentence, pos_tags)
        for sentence, pos_tags in zip(dataset["tokens"], dataset["pos_tags"])
    ]


def preprocess_labels(dataset: Dataset) -> list[list[str]]:
    """Translate raw labels (i.e. integers) to the respective string labels.

    Parameters
    ----------
    dataset : Dataset
        Dataset to preprocess labels.

    Returns
    -------
    list[list[Features]]
        Preprocessed labels.
    """
    labels = dataset.features["ner_tags"].feature.names
    return [[labels[index] for index in indices] for indices in dataset["ner_tags"]]
