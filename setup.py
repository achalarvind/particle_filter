from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "Sensing Package",
    ext_modules = cythonize('sense.pyx')
)
