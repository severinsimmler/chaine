from distutils.command.build_ext import build_ext

from Cython.Build import cythonize
from setuptools import Extension

SOURCES = [
    "chaine/crf.cpp",
    "chaine/crfsuite/lib/cqdb/src/cqdb.c",
    "chaine/crfsuite/lib/cqdb/src/lookup3.c",
    "chaine/crfsuite/lib/crf/src/crf1d_context.c",
    "chaine/crfsuite/lib/crf/src/crf1d_encode.c",
    "chaine/crfsuite/lib/crf/src/crf1d_feature.c",
    "chaine/crfsuite/lib/crf/src/crf1d_model.c",
    "chaine/crfsuite/lib/crf/src/crf1d_tag.c",
    "chaine/crfsuite/lib/crf/src/crfsuite.c",
    "chaine/crfsuite/lib/crf/src/crfsuite_train.c",
    "chaine/crfsuite/lib/crf/src/dataset.c",
    "chaine/crfsuite/lib/crf/src/dictionary.c",
    "chaine/crfsuite/lib/crf/src/holdout.c",
    "chaine/crfsuite/lib/crf/src/logging.c",
    "chaine/crfsuite/lib/crf/src/params.c",
    "chaine/crfsuite/lib/crf/src/quark.c",
    "chaine/crfsuite/lib/crf/src/rumavl.c",
    "chaine/crfsuite/lib/crf/src/train_arow.c",
    "chaine/crfsuite/lib/crf/src/train_averaged_perceptron.c",
    "chaine/crfsuite/lib/crf/src/train_l2sgd.c",
    "chaine/crfsuite/lib/crf/src/train_lbfgs.c",
    "chaine/crfsuite/lib/crf/src/train_passive_aggressive.c",
    "chaine/crfsuite/swig/crfsuite.cpp",
    "chaine/liblbfgs/lib/lbfgs.c",
    "chaine/trainer_wrapper.cpp",
]
INCLUDE_DIRS = [
    "chaine/crfsuite/include/",
    "chaine/crfsuite/lib/cqdb/include",
    "chaine/liblbfgs/include",
    "chaine",
]
EXTENSION = Extension("chaine.crf", language="c++", include_dirs=INCLUDE_DIRS, sources=SOURCES)


class ExtensionBuilder(build_ext):
    def build_extensions(self):
        c = self.compiler
        _compile = c._compile

        def c_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            cc_args = cc_args + ["-std=c99"] if src.endswith(".c") else cc_args
            return _compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        c._compile = c_compile
        build_ext.build_extensions(self)


def build(setup_kwargs: dict):
    # compile source module into C++ files
    cythonize("chaine/crf.pyx", force=True)

    # update setup.py kwargs
    kwargs = {"cmdclass": {"build_ext": ExtensionBuilder}, "ext_modules": [EXTENSION]}
    setup_kwargs.update(kwargs)
