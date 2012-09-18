import os, sys
import numpy
from glob import glob
from setuptools import setup, Extension

_lprmsd = Extension('_lprmsd',
          sources = glob('src/*.c'),
          extra_compile_args = ["-std=c99","-O2",
                                "-msse2","-msse3","-Wno-unused","-fopenmp","-m64"], 
          # If you are 32-bit you should remove the -m64 flag
          extra_link_args = ['-lblas', '-lpthread', '-lm', '-lgomp'],
          include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])

setup(name='msmbuilder.metrics.lprmsd',
      version='1.2',
      py_modules = ['lprmsd'],
      ext_modules = [_lprmsd],
      scripts=glob('scripts/*.py'),
      zip_safe=False,
      entry_points="""
        [msmbuilder.metrics]
         metric_class=lprmsd:LPRMSD
         add_metric_parser=lprmsd:add_metric_parser
         construct_metric=lprmsd:construct_metric
         """)
