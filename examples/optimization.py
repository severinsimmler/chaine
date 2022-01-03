import datasets
from utils import featurize_dataset, preprocess_labels

from chaine import Optimizer
from chaine.logging import Logger

LOGGER = Logger(__name__)


if __name__ == "__main__":
    LOGGER.info("Loading raw data set")
    dataset = datasets.load_dataset("conll2003")

    LOGGER.info(f"Number of sentences for training: {len(dataset['train']['tokens'])}")

    LOGGER.info("Extracting features from train set for optimization")
    sentences = featurize_dataset(dataset["train"])
    labels = preprocess_labels(dataset["train"])

    LOGGER.info("Start optimization with downsampled data set")
    result = Optimizer().optimize_hyperparameters(sentences, labels, sample_size=1000)

    LOGGER.info(f"Best result: {result[0]}")
