# Copyright 2017 Hieu Tran. All Rights Reserved.
#
# DeepNovo is publicly available for non-commercial uses.
# ==============================================================================

"""TODO(nh2tran): docstring."""

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("deepnovo_cython_modules.pyx"))
