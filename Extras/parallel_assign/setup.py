import sys
import glob
from setuptools import setup

setup(name='parallel_assign',
      version = '1.0',
      description = 'Parallel Assignment MSMBuilder',
      packages=['parallel_assign',
                'parallel_assign.scripts'],
      package_dir={'parallel_assign':'lib',
                   'parallel_assign.scripts':'scripts'},
      install_requires=['ipython==0.13'],
      scripts=filter(lambda elem: '_' not in elem, glob.glob('scripts/*')))
