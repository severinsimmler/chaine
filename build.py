import glob
import subprocess

from setuptools import Extension
from distutils.command.build_ext import build_ext as _build_ext


sources = ["chaine/model.cpp", "chaine/trainer_wrapper.cpp"]
sources += glob.glob("chaine/crfsuite/lib/crf/src/*.c")
sources += glob.glob("chaine/crfsuite/swig/*.cpp")
sources += ["chaine/crfsuite/lib/cqdb/src/cqdb.c"]
sources += ["chaine/crfsuite/lib/cqdb/src/lookup3.c"]
sources += glob.glob("chaine/liblbfgs/lib/*.c")
sources = sorted(sources)


includes = [
    "chaine/crfsuite/include/",
    "chaine/crfsuite/lib/cqdb/include",
    "chaine/liblbfgs/include",
    "chaine",
]


class build_ext(_build_ext):
    def build_extensions(self):
        c = self.compiler
        _compile = c._compile

        def c_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            cc_args = cc_args + ["-std=c99"] if src.endswith(".c") else cc_args
            return _compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        c._compile = c_compile
        _build_ext.build_extensions(self)


ext_modules = [Extension("chaine.model", include_dirs=includes, language="c++", sources=sources)]


def build(setup_kwargs):
    # cythonize
    subprocess.check_call(["cython", "chaine/model.pyx", "--cplus", "-2", "-I", "chaine"])

    # update setup.py kwargs
    kwargs = {"cmdclass": {"build_ext": build_ext}, "ext_modules": ext_modules}
    setup_kwargs.update(kwargs)
