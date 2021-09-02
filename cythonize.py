from Cython.Build import cythonize


if __name__ == "__main__":
    cythonize("chaine/_core/crf.pyx", force=True)
