import datasets
from seqeval.metrics import classification_report
from utils import featurize_dataset, preprocess_labels

import chaine
from chaine.logging import Logger

LOGGER = Logger(__name__)

if __name__ == "__main__":
    LOGGER.info("Loading raw data set")
    dataset = datasets.load_dataset("conll2003")

    LOGGER.info(f"Number of sentences for training: {len(dataset['train']['tokens'])}")
    LOGGER.info(f"Number of sentences for evaluation: {len(dataset['test']['tokens'])}")

    LOGGER.info("Extracting features from train set for training")
    sentences = featurize_dataset(dataset["train"])
    labels = preprocess_labels(dataset["train"])

    LOGGER.info("Starting training")
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

    LOGGER.info("Extracting features from test set for evaluation")
    sentences = featurize_dataset(dataset["test"])
    labels = preprocess_labels(dataset["test"])

    LOGGER.info("Evaluating the trained model")
    predictions = model.predict(sentences)

    print("\n", classification_report(labels, predictions))
