import glob
import sys

from setuptools import setup, Extension
from distutils.command.build_ext import build_ext

sources = ["chaine/model.cpp", "chaine/trainer_wrapper.cpp"]

# crfsuite
sources += glob.glob("chaine/crfsuite/lib/crf/src/*.c")
sources += glob.glob("chaine/crfsuite/swig/*.cpp")

sources += ["chaine/crfsuite/lib/cqdb/src/cqdb.c"]
sources += ["chaine/crfsuite/lib/cqdb/src/lookup3.c"]

# lbfgs
sources += glob.glob("chaine/liblbfgs/lib/*.c")

includes = [
    "chaine/crfsuite/include/",
    "chaine/crfsuite/lib/cqdb/include",
    "chaine/liblbfgs/include",
    "chaine",
]


class build_ext_check_gcc(build_ext):
    def build_extensions(self):
        c = self.compiler

        _compile = c._compile

        def c_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            cc_args = cc_args + ["-std=c99"] if src.endswith(".c") else cc_args
            return _compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        if c.compiler_type == "unix" and "gcc" in c.compiler:
            c._compile = c_compile
        
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "chaine.model", include_dirs=includes, language="c++", sources=sorted(sources)
    )
]


setup(
    name="chaine",
    version="1.0.0",
    description="A lightweight Linear-Chain Conditional Random Field",
    zip_safe=False,
    packages=["chaine"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext_check_gcc},
)
