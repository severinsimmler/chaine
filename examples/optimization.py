import json

import datasets
from utils import featurize_dataset, preprocess_labels

from chaine import Optimizer
from chaine.logging import Logger
from chaine.optimization import LBFGSSearchSpace

LOGGER = Logger(__name__)


if __name__ == "__main__":
    LOGGER.info("Loading raw dataset")
    dataset = datasets.load_dataset("conll2003")

    LOGGER.info(f"Number of sentences for training: {len(dataset['train']['tokens'])}")
    LOGGER.info(f"Number of sentences for evaluation: {len(dataset['test']['tokens'])}")

    LOGGER.info("Extracting features from dataset for training")
    sentences = featurize_dataset(dataset["train"])
    labels = preprocess_labels(dataset["train"])

    optimizer = Optimizer()
    result = optimizer.optimize(sentences, labels, sample_size=1000)

    with open("hyperparameter-optimization.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
