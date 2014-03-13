"""MSMBuilder: a python library for Markov state models of conformational dynamics

MSMBuilder (https://simtk.org/home/msmbuilder)
is a library that provides tools for analyzing molecular dynamics
simulations, particularly through the construction
of Markov state models for conformational dynamics.
"""

from __future__ import print_function
DOCLINES = __doc__.split("\n")

import os
import sys
import subprocess
from glob import glob
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

try:
    import numpy
    import scipy
    if not hasattr(numpy.version, 'full_version') or numpy.version.full_version < '1.6':
        raise ImportError()
    if not hasattr(scipy.version, 'full_version') or scipy.version.full_version < '0.11':
        raise ImportError()
except ImportError:
    print('numpy>=1.6 and scipy>=0.11 are required for msmbuilder', file=sys.stderr)
    sys.exit(1)

VERSION = "2.8"
ISRELEASED = False
__author__ = "MSMBuilder Team"
__version__ = VERSION

# metadata for setup()
metadata = {
    'name': 'msmbuilder',
    'version': VERSION,
    'author': __author__,
    'author_email': 'msmbuilder-user@stanford.edu',
    'license': 'GPL v3.0',
    'url': 'https://simtk.org/home/msmbuilder',
    'download_url': 'https://simtk.org/home/msmbuilder',
    'platforms': ["Linux", "Mac OS X"],
    'description': DOCLINES[0],
    'long_description':"\n".join(DOCLINES[2:]),
    'packages': ['msmbuilder', 'msmbuilder.scripts', 'msmbuilder.project',
                 'msmbuilder.lumping', 'msmbuilder.geometry',
                 'msmbuilder.metrics', 'msmbuilder.reduce',
                 'msmbuilder.reference'],
    'package_dir': {'msmbuilder': 'src/python', 'msmbuilder.scripts': 'scripts',
                    'msmbuilder.reference': 'reference'},
    'package_data': {'msmbuilder.reference': [os.path.relpath(os.path.join(a[0], b), 'reference') for a in os.walk('reference') for b in a[2]]},
    'scripts': ['scripts/msmb'] + [e for e in glob('scripts/*') if not e.endswith('__.py')]
}



# Return the git revision as a string
# copied from numpy setup.py
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def write_version_py(filename='src/python/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM MSMBUILDER SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version' : FULLVERSION,
                       'git_revision' : GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


write_version_py()
setup(**metadata)

