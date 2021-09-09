# cython: embedsignature=True
# cython: c_string_type=str
# cython: c_string_encoding=utf-8
# cython: profile=False
# cython: language_level=2
# distutils: language=c++

cimport crfsuite_api
from libcpp.string cimport string
import os

from chaine.logging import Logger
from chaine.typing import Dict, Iterable, List, Filepath, Sequence

LOGGER = Logger(__name__)


cdef class Trainer:
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

    @property
    def _params(self):
        return self._c_trainer.params()

    def _train(self, model_filepath: Filepath):
        self._c_trainer.train(str(model_filepath), -1)

    def _log(self):
        raise NotImplemented("You have to implement this method.")

    cdef _on_message(self, string message):
        self._log(message)

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
    cdef crfsuite_api.Tagger c_tagger

    def __init__(self, model_filepath: Filepath):
        self._load(model_filepath)

    @property
    def _labels(self):
        return self.c_tagger.labels()

    def _predict_single(self, sequence: Sequence) -> List[str]:
        self._set_sequence(sequence)
        return self.c_tagger.viterbi()

    def _predict_proba_single(self, sequence: Sequence) -> List[Dict[str, float]]:
        self._set_sequence(sequence)
        return [
            {label: self._marginal(label, index) for label in self._labels}
            for index in range(len(sequence))
        ]

    def _load(self, filepath: Filepath):
        filepath = str(filepath)
        self._check_model(filepath)
        if not self.c_tagger.open(filepath):
            raise ValueError(f"Cannot load model file {filepath}")

    def _marginal(self, label: str, index: int):
        return self.c_tagger.marginal(label, index)

    cpdef _set_sequence(self, sequence) except +:
        self.c_tagger.set(to_seq(sequence))

    @staticmethod
    def _check_model(filepath: str):
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

    if isinstance(sequence, ItemSequence):
        c_sequence = (<ItemSequence>sequence).c_sequence
    else:
        for s in sequence:
            c_sequence.push_back(to_item(s))
    return c_sequence


cdef class ItemSequence:
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
        return f"<ItemSequence ({len(self)})>"
