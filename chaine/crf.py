"""
chaine.crf
~~~~~~~~~~

This module implements the trainer and model.
"""

import json

from chaine.core.crf import Model as _Model
from chaine.core.crf import Trainer as _Trainer
from chaine.logging import Logger
from chaine.typing import Dict, Filepath, Iterable, Labels, List, Sequence

LOGGER = Logger(__name__)


class Trainer(_Trainer):
    """Trainer for conditional random fields.

    Parameters
    ----------
    algorithm : str
        The following optimization algorithms are available:
            * lbfgs: Limited-memory BFGS with L1/L2 regularization
            * l2sgd: Stochastic gradient descent with L2 regularization
            * ap: Averaged perceptron
            * pa: Passive aggressive
            * arow: Adaptive regularization of weights

    Limited-memory BFGS Parameters (lbfgs)
    --------------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (unlimited by default).
    num_memories : int, optional (default=6)
        Number of limited memories for approximating the inverse hessian matrix.
    c1 : float, optional (default=0)
        Coefficient for L1 regularization.
    c2 : float, optional (default=1.0)
        Coefficient for L2 regularization.
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence.
    period : int, optional (default=10)
        Threshold value for iterations to test the stopping criterion.
    delta : float, optional (default=1e-5)
        Top iteration when log likelihood is not greater than this.
    linesearch : str, optional (default="MoreThuente")
        Line search algorithm used in updates:
            * MoreThuente: More and Thuente's method
            * Backtracking: Backtracking method with regular Wolfe condition
            * StrongBacktracking: Backtracking method with strong Wolfe condition
    max_linesearch : int, optional (default=20)
        Maximum number of trials for the line search algorithm.

    SGD with L2 Parameters (l2sgd)
    ------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (1000 by default).
    c2 : float, optional (default=1.0)
        Coefficient for L2 regularization.
    period : int, optional (default=10)
        Threshold value for iterations to test the stopping criterion.
    delta : float, optional (default=1e-5)
        Top iteration when log likelihood is not greater than this.
    calibration_eta : float, optional (default=0.1)
        Initial value of learning rate (eta) used for calibration.
    calibration_rate : float, optional (default=2.0)
        Rate of increase/decrease of learning rate for calibration.
    calibration_samples : int, optional (default=1000)
        Number of instances used for calibration.
    calibration_candidates : int, optional (default=10)
        Number of candidates of learning rate.
    calibration_max_trials : int, optional (default=20)
        Maximum number of trials of learning rates for calibration.

    Averaged Perceptron Parameters (ap)
    -----------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default).
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence.

    Passive Aggressive Parameters (pa)
    ----------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default).
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence.
    pa_type : int, optional (default=1)
        Strategy for updating feature weights:
            * 0: PA without slack variables
            * 1: PA type I
            * 2: PA type II
    c : float, optional (default=1)
        Aggressiveness parameter (used only for PA-I and PA-II).
    error_sensitive : bool, optional (default=True)
        Include square root of predicted incorrect labels into optimization routine.
    averaging : bool, optional (default=True)
        Compute average of feature weights at all updates.

    Adaptive Regularization of Weights Parameters (arow)
    ----------------------------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data.
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data.
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data.
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default).
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence.
    variance : float, optional (default=1)
        Initial variance of every feature weight.
    gamma : float, optional (default=1)
        Trade-off between loss function and changes of feature weights.
    """

    def __init__(self, algorithm: str = "l2sgd", **kwargs):
        self.algorithm = algorithm
        super().__init__(algorithm, **kwargs)

    def __repr__(self):
        return f"<Trainer ({self.algorithm}): {self.params}>"

    def train(
        self,
        dataset: Iterable[Sequence],
        labels: Iterable[Labels],
        model_filepath: Filepath,
    ):
        LOGGER.info("Loading training data")
        for i, (sequence, labels_) in enumerate(zip(dataset, labels)):
            # log progress every 100 data points
            if i > 0 and i % 100 == 0:
                LOGGER.debug(f"{i} processed data points")

            try:
                self._append(sequence, labels_)
            except Exception as message:
                LOGGER.error(message)
                LOGGER.debug(f"Sequence: {json.dumps(sequence)}")
                LOGGER.debug(f"Labels: {json.dumps(labels_)}")

        # fire!
        self._train(model_filepath)

    @property
    def params(self):
        return {
            self._param2kwarg.get(name, name): self._get_param(name)
            for name in self._params
        }

    def _log(self, message):
        LOGGER.info(message)


class Model(_Model):
    """Linear-chain conditional random field.

    Parameters
    ----------
    model_filepath : Filepath
        Path to the trained model.
    """

    def __repr__(self):
        """Representation of the model"""
        return f"<Model: {self.labels}>"

    @property
    def labels(self):
        """Labels the model is trained on"""
        return set(self._labels)

    def predict_single(self, sequence: Sequence) -> List[str]:
        """Predict most likely labels for a given sequence of features

        Parameters
        ----------
        sequence : Sequence
            Sequence of features, e.g. [{"feature_a", "feature_b"}, {"feature_c"}]

        Returns
        -------
        List[str]
            Most likely label sequence
        """
        return self._predict_single(sequence)

    def predict(self, sequences: Iterable[Sequence]) -> List[List[str]]:
        """Predict most likely labels for a batch of sequences

        Parameters
        ----------
        sequences : Iterable[Sequence]
            Batch of sequences

        Returns
        -------
        List[List[str]]
            Most likely label sequences
        """
        return [self.predict_single(sequence) for sequence in sequences]

    def predict_proba_single(self, sequence: Sequence) -> List[Dict[str, float]]:
        """Predict probabilities over all labels for each token in a sequence

        Parameters
        ----------
        sequence : Sequence
            Sequence of features, e.g. [{"feature_a", "feature_b"}, {"feature_c"}]

        Returns
        -------
        List[Dict[str, float]]
            Probability distributions over all labels for each token
        """
        if not isinstance(sequence, list):
            sequence = list(sequence)
        return self._predict_proba_single(sequence)

    def predict_proba(
        self, sequences: Iterable[Sequence]
    ) -> List[List[Dict[str, float]]]:
        """Predict probabilities over all labels for each token in a batch of sequences

        Parameters
        ----------
        sequences : Sequence
            Batch of sequences

        Returns
        -------
        List[Dict[str, float]]
            Probability distributions over all labels for each token in the sequences
        """
        return [self.predict_proba_single(sequence) for sequence in sequences]