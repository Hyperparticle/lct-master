# https://www.scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html#id13
# $ python setup.py build_ext -i

from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("cos_doubles",
                 sources=["_cos_doubles.pyx", "cos_doubles.c"],
                 include_dirs=[numpy.get_include()])],
)
