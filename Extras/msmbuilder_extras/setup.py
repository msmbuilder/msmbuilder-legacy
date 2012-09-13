import sys
import glob
from setuptools import setup

setup(name='msmbuild_extras',
      version = '2.6.dev',
      description = 'Various extras. These are in various states of disrepair',
      packages=['msmbuilder_extras'],
      package_dir={'msmbuilder_extras':'lib'},
      scripts=glob.glob('scripts/*.py'))
