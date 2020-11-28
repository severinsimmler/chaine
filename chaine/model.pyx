# cython: embedsignature=True
# cython: c_string_type=str
# cython: c_string_encoding=utf-8
# cython: profile=False

cimport crfsuite_api
from libcpp.string cimport string
import os

from chaine.utils import _LogParser


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
        for x in sequence:
            c_sequence.push_back(to_item(x))
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


def intbool(value):
    return bool(int(value))


cdef class Trainer:
    cdef crfsuite_api.Trainer _c_trainer

    _algorithm_aliases = {
            "ap": "averaged-perceptron",
            "pa": "passive-aggressive",
            "lbfgs": "lbfgs"
        }
    _parameter_types = {
            "feature.minfreq": float,
            "feature.possible_states": intbool,
            "feature.possible_transitions": intbool,
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
            "error_sensitive": intbool,
            "averaging": intbool,
            "variance": float,
            "gamma": float,
        }
    _log_parser = _LogParser()

    def __init__(self, algorithm, **params):
        self._select_algorithm(algorithm)
        self._set_params(dict(params))

    def __cinit__(self):
        self._c_trainer.set_handler(self, <crfsuite_api.messagefunc>self._on_message)
        self._c_trainer.select("lbfgs", "crf1d")
        self._c_trainer._init_trainer()

    def __repr__(self):
        return f"<Trainer: {self.params}>"

    cdef _on_message(self, string message):
        self._message(message)

    def _message(self, message):
        event = self._log_parser.parse(message)
        if event:
            # TODO replace with proper logging
            print(event)

    def _append(self, sequence, labels, int group=0):
        self._c_trainer.append(to_seq(sequence), labels, group)

    def _select_algorithm(self, algorithm):
        algorithm = self._algorithm_aliases[algorithm.lower()]
        if not self._c_trainer.select(algorithm, "crf1d"):
            raise ValueError(f"Bad arguments: algorithm={algorithm}")

    def train(self, X, y, model_filepath, int holdout=-1):
        # TODO replace with proper logging
        print("Loading data")
        for sequence, labels in zip(X, y):
            self._append(sequence, labels)
        print("Start training")
        status_code = self._c_trainer.train(model_filepath, holdout)
        if status_code != crfsuite_api.CRFSUITE_SUCCESS:
            # TODO replace with proper error handling
            print(status_code)

    @property
    def params(self):
        return {name: self._get_param(name) for name in self._c_trainer.params()}

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


cdef class CRF:
    cdef crfsuite_api.Tagger c_tagger

    def __init__(self, model_filepath):
        self._load(model_filepath)

    def __repr__(self):
        return f"<CRF: {self.labels}>"

    def _load(self, filepath):
        self._check_model(filepath)
        if not self.c_tagger.open(filepath):
            raise ValueError(f"Cannot load model file {filepath}")

    @property
    def labels(self):
        return set(self.c_tagger.labels())

    def predict(self, sequence):
        self._set_sequence(sequence)
        return self.c_tagger.viterbi()

    def _marginal(self, label, index):
        return self.c_tagger.marginal(label, index)

    def predict_marginals_single(self, sequence):
        self._set_sequence(sequence)
        return [
            {label: self._marginal(label, index) for label in self.labels}
            for index in range(len(sequence))
        ]

    def predict_marginals(self, sequences):
        return [self.predict_marginals_single(sequence) for sequence in sequences]

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
