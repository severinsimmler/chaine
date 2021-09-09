from Cython.Build import cythonize


if __name__ == "__main__":
    cythonize("chaine/core/crf.pyx", force=True)
