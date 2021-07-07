# cython: embedsignature=True
# cython: c_string_type=str
# cython: c_string_encoding=utf-8
# cython: profile=False

cimport crfsuite_api
from libcpp.string cimport string
import json
import os

from chaine.logging import Logger
from chaine.typing import Dataset, Dict, Iterable, Labels, List, Filepath, Sequence

LOGGER = Logger(__name__)


cdef class Trainer:
    """Model trainer

    Parameters
    ----------
    algorithm : str
        Following algorithms are available:
            * lbfgs: Limited-memory BFGS with L1/L2 regularization
            * l2sgd: Stochastic gradient descent with L2 regularization
            * ap: Averaged perceptron
            * pa: Passive aggressive
            * arow: Adaptive regularization of weights

    Limited-memory BFGS Parameters
    ------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data
    max_iterations : int, optional (default=None)
        Maximum number of iterations (unlimited by default)
    num_memories : int, optional (default=6)
        Number of limited memories for approximating the inverse hessian matrix
    c1 : float, optional (default=0)
        Coefficient for L1 regularization
    c2 : float, optional (default=1.0)
        Coefficient for L2 regularization
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence
    period : int, optional (default=10)
        Threshold value for iterations to test the stopping criterion
    delta : float, optional (default=1e-5)
        Top iteration when log likelihood is not greater than this
    linesearch : str, optional (default="MoreThuente")
        Line search algorithm used in updates:
            * MoreThuente: More and Thuente's method
            * Backtracking: Backtracking method with regular Wolfe condition
            * StrongBacktracking: Backtracking method with strong Wolfe condition
    max_linesearch : int, optional (default=20)
        Maximum number of trials for the line search algorithm

    SGD with L2 Parameters
    ----------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data
    max_iterations : int, optional (default=None)
        Maximum number of iterations (1000 by default)
    c2 : float, optional (default=1.0)
        Coefficient for L2 regularization
    period : int, optional (default=10)
        Threshold value for iterations to test the stopping criterion
    delta : float, optional (default=1e-5)
        Top iteration when log likelihood is not greater than this
    calibration_eta : float, optional (default=0.1)
        Initial value of learning rate (eta) used for calibration
    calibration_rate : float, optional (default=2.0)
        Rate of increase/decrease of learning rate for calibration
    calibration_samples : int, optional (default=1000)
        Number of instances used for calibration
    calibration_candidates : int, optional (default=10)
        Number of candidates of learning rate
    calibration_max_trials : int, optional (default=20)
        Maximum number of trials of learning rates for calibration

    Averaged Perceptron Parameters
    ------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default)
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence

    Passive Aggressive Parameters
    -----------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default)
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence
    pa_type : int, optional (default=1)
        Strategy for updating feature weights:
            * 0: PA without slack variables
            * 1: PA type I
            * 2: PA type II
    c : float, optional (default=1)
        Aggressiveness parameter (used only for PA-I and PA-II)
    error_sensitive : bool, optional (default=True)
        Include square root of predicted incorrect labels into optimization routine
    averaging : bool, optional (default=True)
        Compute average of feature weights at all updates

    Adaptive Regularization of Weights (AROW) Parameters
    ----------------------------------------------------
    min_freq : float, optional (default=0)
        Threshold value for minimum frequency of a feature occurring in training data
    all_possible_states : bool, optional (default=False)
        Generate state features that do not even occur in the training data
    all_possible_transitions : bool, optional (default=False)
        Generate transition features that do not even occur in the training data
    max_iterations : int, optional (default=None)
        Maximum number of iterations (100 by default)
    epsilon : float, optional (default=1e-5)
        Parameter that determines the condition of convergence
    variance : float, optional (default=1)
        Initial variance of every feature weight
    gamma : float, optional (default=1)
        Trade-off between loss function and changes of feature weights
    """
    cdef crfsuite_api.Trainer _c_trainer

    _algorithm_aliases = {
        "lbfgs": "lbfgs",
        "limited-memory-bfgs": "lbfgs",
        "l2sgd": "l2sgd",
        "sgd": "l2sgd",
        "stochastic-gradient-descent": "l2sgd",
        "ap": "averaged-perceptron",
        "averaged-perceptron": "averaged-perceptron",
        "pa": "passive-aggressive",
        "passive-aggressive": "passive-aggressive",
        "arow": "arow"
    }
    _param2kwarg = {
        "feature.minfreq": "min_freq",
        "feature.possible_states": "all_possible_states",
        "feature.possible_transitions": "all_possible_transitions",
        "calibration.eta": "calibration_eta",
        "calibration.rate": "calibration_rate",
        "calibration.samples": "calibration_samples",
        "calibration.candidates": "calibration_candidates",
        "calibration.max_trials": "calibration_max_trials",
        "type": "pa_type",
    }
    _kwarg2param = {
        "min_freq": "feature.minfreq",
        "all_possible_states": "feature.possible_states",
        "all_possible_transitions": "feature.possible_transitions",
        "calibration_eta": "calibration.eta",
        "calibration_rate": "calibration.rate",
        "calibration_samples": "calibration.samples",
        "calibration_candidates": "calibration.candidates",
        "calibration_max_trials": "calibration.max_trials",
        "pa_type": "type",
    }
    _parameter_types = {
            "feature.minfreq": float,
            "feature.possible_states": lambda value: bool(int(value)),
            "feature.possible_transitions": lambda value: bool(int(value)),
            "c1": float,
            "c2": float,
            "max_iterations": int,
            "num_memories": int,
            "epsilon": float,
            "period": int,
            "delta": float,
            "linesearch": str,
            "max_linesearch": int,
            "calibration.eta": float,
            "calibration.rate": float,
            "calibration.samples": float,
            "calibration.candidates": int,
            "calibration.max_trials": int,
            "type": int,
            "c": float,
            "error_sensitive": lambda value: bool(int(value)),
            "averaging": lambda value: bool(int(value)),
            "variance": float,
            "gamma": float,
        }

    def __init__(self, algorithm="l2sgd", **kwargs):
        self._select_algorithm(algorithm)
        params = self._translate_params(kwargs)
        self._set_params(params)

    def __cinit__(self):
        self._c_trainer.set_handler(self, <crfsuite_api.messagefunc>self._on_message)
        self._c_trainer.select("l2sgd", "crf1d")
        self._c_trainer._init_trainer()

    def __repr__(self):
        """Representation of the trainer"""
        return f"<Trainer: {self.params}>"

    def train(self, dataset: Dataset, labels: Labels, model_filepath: Filepath):
        """Train a conditional random field

        Parameters
        ----------
        dataset : Dataset
            Training data set
        labels : Labels
            Corresponding true labels
        model_filepath : Filepath
            Path the trained model is written to

        Note
        ----
        A dataset is three-dimensional:

            [[['feature_a', 'feature_b'],
              ['feature_c']]]

        An instance of this dataset is two-dimensional:

            [['feature_a', 'feature_b'],
             ['feature_c']]

        An item of this instance represents e.g. one word in a sentence by descriptive
        features. One item consists only of the relevant features. Internally, the
        string features are hash-mapped and a sparse matrix is constructed.
        """
        LOGGER.info("Loading training data (this may take a while)")
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

        status_code = self._c_trainer.train(str(model_filepath), -1)
        if status_code != crfsuite_api.CRFSUITE_SUCCESS:
            LOGGER.error(f"An error ({status_code}) occured")

    @property
    def params(self):
        """Training parameters"""
        return {
            self._param2kwarg.get(name, name): self._get_param(name)
            for name in self._c_trainer.params()
        }

    cdef _on_message(self, string message):
        self._message(message)

    def _message(self, message):
        LOGGER.info(message)

    def _append(self, sequence, labels, int group=0):
        # no generators allowed
        if not isinstance(sequence, list):
            sequence = [item for item in sequence]
        if not isinstance(labels, list):
            labels = [label for label in labels]
        # labels must be strings
        labels = [str(label) for label in labels]
        self._c_trainer.append(to_seq(sequence), labels, group)

    def _translate_params(self, kwargs):
        return {
            self._kwarg2param.get(kwarg, kwarg): value
            for kwarg, value in kwargs.items()
        }

    def _select_algorithm(self, algorithm):
        algorithm = self._algorithm_aliases[algorithm.lower()]
        if not self._c_trainer.select(algorithm, "crf1d"):
            raise ValueError(f"{algorithm} is no available algorithm")

    def _set_params(self, params):
        for param, value in params.items():
            self._set_param(param, value)

    def _set_param(self, param, value):
        if isinstance(value, bool):
            value = int(value)
        self._c_trainer.set(param, str(value))

    def _get_param(self, param):
        return self._cast_parameter(param, self._c_trainer.get(param))

    def _cast_parameter(self, param, value):
        if param in self._parameter_types:
            return self._parameter_types[param](value)
        return value


cdef class Model:
    """Linear-chain conditional random field

    Parameters
    ----------
    model_filepath : Filepath
        Path to the trained model
    """
    cdef crfsuite_api.Tagger c_tagger

    def __init__(self, model_filepath: Filepath):
        self._load(str(model_filepath))

    def __repr__(self):
        """Representation of the model"""
        return f"<CRF: {self.labels}>"

    @property
    def labels(self):
        """Labels the model is trained on"""
        return set(self.c_tagger.labels())

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
        self._set_sequence(sequence)
        return self.c_tagger.viterbi()

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
        self._set_sequence(sequence)
        return [
            {label: self._marginal(label, index) for label in self.labels}
            for index in range(len(sequence))
        ]

    def predict_proba(self, sequences: Iterable[Sequence]) -> List[List[Dict[str, float]]]:
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

    def _load(self, filepath):
        filepath = str(filepath)
        self._check_model(filepath)
        if not self.c_tagger.open(filepath):
            raise ValueError(f"Cannot load model file {filepath}")

    def _marginal(self, label, index):
        return self.c_tagger.marginal(label, index)

    cpdef _set_sequence(self, sequence) except +:
        self.c_tagger.set(to_seq(sequence))

    @staticmethod
    def _check_model(filepath):
        with open(filepath, "rb") as model:
            magic = model.read(4)
            if magic != b"lCRF":
                raise ValueError(f"Invalid model file {filepath}")
            model.seek(0, os.SEEK_END)
            if model.tell() <= 48:
                raise ValueError(f"Model file {filepath} does not have a complete header")


cdef crfsuite_api.Item to_item(sequence) except+:
    cdef crfsuite_api.Item c_item
    cdef double c_value
    cdef string c_token
    cdef string separator
    cdef bint is_dict, is_nested_value

    separator = b":"
    is_dict = isinstance(sequence, dict)
    c_item = crfsuite_api.Item()
    c_item.reserve(len(sequence))

    for token in sequence:
        if isinstance(token, unicode):
            c_token = (<unicode>token).encode("utf8")
        else:
            c_token = token
        if not is_dict:
            c_value = 1.0
            c_item.push_back(crfsuite_api.Attribute(c_token, c_value))
        else:
            value = (<dict>sequence)[token]
            if isinstance(value, (dict, list, set)):
                for attr in to_item(value):
                    c_item.push_back(
                        crfsuite_api.Attribute(c_token + separator + attr.attr, attr.value)
                    )
            else:
                if isinstance(value, unicode):
                    c_token += separator
                    c_token += <string>(<unicode>value).encode("utf8")
                    c_value = 1.0
                elif isinstance(value, bytes):
                    c_token += separator
                    c_token += <string>value
                    c_value = 1.0
                else:
                    c_value = value
                c_item.push_back(crfsuite_api.Attribute(c_token, c_value))
    return c_item


cdef crfsuite_api.ItemSequence to_seq(sequence) except+:
    cdef crfsuite_api.ItemSequence c_sequence

    if isinstance(sequence, _ItemSequence):
        c_sequence = (<_ItemSequence>sequence).c_sequence
    else:
        for s in sequence:
            c_sequence.push_back(to_item(s))
    return c_sequence


cdef class _ItemSequence:
    cdef crfsuite_api.ItemSequence c_sequence

    def __init__(self, sequence):
        self.c_sequence = to_seq(sequence)

    def items(self):
        cdef crfsuite_api.Item c_item
        cdef crfsuite_api.Attribute c_attr
        cdef bytes token

        sequence = []
        for c_item in self.c_sequence:
            x = {}
            for c_attr in c_item:
                token = <bytes>c_attr.attr.c_str()
                x[token.decode("utf8")] = c_attr.value
            sequence.append(x)
        return sequence

    def __len__(self):
        return self.c_sequence.size()

    def __repr__(self):
        return f"<_ItemSequence ({len(self)})>"
