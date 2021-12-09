from distutils.command.build_ext import build_ext

from setuptools import Extension

SOURCES = [
    "chaine/core/crf.cpp",
    "chaine/core/crfsuite/lib/cqdb/src/cqdb.c",
    "chaine/core/crfsuite/lib/cqdb/src/lookup3.c",
    "chaine/core/crfsuite/lib/crf/src/crf1d_context.c",
    "chaine/core/crfsuite/lib/crf/src/crf1d_encode.c",
    "chaine/core/crfsuite/lib/crf/src/crf1d_feature.c",
    "chaine/core/crfsuite/lib/crf/src/crf1d_model.c",
    "chaine/core/crfsuite/lib/crf/src/crf1d_tag.c",
    "chaine/core/crfsuite/lib/crf/src/crfsuite.c",
    "chaine/core/crfsuite/lib/crf/src/crfsuite_train.c",
    "chaine/core/crfsuite/lib/crf/src/dataset.c",
    "chaine/core/crfsuite/lib/crf/src/dictionary.c",
    "chaine/core/crfsuite/lib/crf/src/holdout.c",
    "chaine/core/crfsuite/lib/crf/src/json.c",
    "chaine/core/crfsuite/lib/crf/src/logging.c",
    "chaine/core/crfsuite/lib/crf/src/params.c",
    "chaine/core/crfsuite/lib/crf/src/quark.c",
    "chaine/core/crfsuite/lib/crf/src/rumavl.c",
    "chaine/core/crfsuite/lib/crf/src/train_arow.c",
    "chaine/core/crfsuite/lib/crf/src/train_averaged_perceptron.c",
    "chaine/core/crfsuite/lib/crf/src/train_l2sgd.c",
    "chaine/core/crfsuite/lib/crf/src/train_lbfgs.c",
    "chaine/core/crfsuite/lib/crf/src/train_passive_aggressive.c",
    "chaine/core/crfsuite/swig/crfsuite.cpp",
    "chaine/core/liblbfgs/lib/lbfgs.c",
    "chaine/core/trainer_wrapper.cpp",
]
INCLUDE_DIRS = [
    "chaine/core/crfsuite/include/",
    "chaine/core/crfsuite/lib/cqdb/include",
    "chaine/core/liblbfgs/include",
    "chaine/core",
]
EXTENSION = Extension("chaine.core.crf", language="c++", include_dirs=INCLUDE_DIRS, sources=SOURCES)


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
    setup_kwargs.update({"cmdclass": {"build_ext": ExtensionBuilder}, "ext_modules": [EXTENSION]})
