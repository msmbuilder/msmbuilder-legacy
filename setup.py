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
import shutil
import tempfile
import subprocess
from glob import glob
from distutils.ccompiler import new_compiler
from setuptools import setup, Extension

try:
    import scipy
    if scipy.version.version < '0.11':
        raise ImportError()
except ImportError:
    print('scipy version 0.11 or better is required for msmbuilder', file=sys.stderr)
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
    'zip_safe': False,
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


def hasfunction(cc, funcname, include=None, extra_postargs=None):
    # From http://stackoverflow.com/questions/
    #            7018879/disabling-output-when-compiling-with-distutils
    tmpdir = tempfile.mkdtemp(prefix='hasfunction-')
    devnull = oldstderr = None
    try:
        try:
            fname = os.path.join(tmpdir, 'funcname.c')
            f = open(fname, 'w')
            if include is not None:
                f.write('#include %s\n' % include)
            f.write('int main(void) {\n')
            f.write('    %s;\n' % funcname)
            f.write('}\n')
            f.close()
            devnull = open(os.devnull, 'w')
            oldstderr = os.dup(sys.stderr.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            objects = cc.compile([fname], output_dir=tmpdir,
                                 extra_postargs=extra_postargs)
            cc.link_executable(objects, os.path.join(tmpdir, 'a.out'))
        except Exception as e:
            return False
        return True
    finally:
        if oldstderr is not None:
            os.dup2(oldstderr, sys.stderr.fileno())
        if devnull is not None:
            devnull.close()
        shutil.rmtree(tmpdir)


def detect_openmp():
    "Does this compiler support OpenMP parallelization?"
    compiler = new_compiler()
    print('Attempting to autodetect OpenMP support...')
    hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
    needs_gomp = hasopenmp
    if not hasopenmp:
        compiler.add_library('gomp')
        hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
        needs_gomp = hasopenmp
    print
    if hasopenmp:
        print('Compiler supports OpenMP')
    else:
        print('Did not detect OpenMP support; parallel RMSD disabled')
    return hasopenmp, needs_gomp

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

def travis_main():
    # On READTHEDOCS, the service that hosts our documentation, the build
    # environment does not have numpy and cannot build C extension modules,
    # so if we detect this environment variable, we're going to bail out
    # and run a minimal setup. This only installs the python packages, which
    # is not enough to RUN anything, but should be enough to introspect the
    # docstrings, which is what's needed for the documentation
    from distutils.core import setup
    import tempfile, shutil
    write_version_py()

    # dirty, dirty trick to install "mock" packages
    mockdir = tempfile.mkdtemp()
    open(os.path.join(mockdir, '__init__.py'), 'w').close()
    extensions = ['msmbuilder._distance_wrap', 'msmbuilder._contact_wrap']
    metadata['packages'].extend(extensions)
    for ex in extensions:
        metadata['package_dir'][ex] = mockdir
    # end dirty trick :)

    setup(**metadata)
    shutil.rmtree(mockdir) #clean up dirty trick

def main():
    import numpy
    import setuptools

    compiler_args = ['-O3', '-funroll-loops']
    if new_compiler().compiler_type == 'msvc':
        compiler_args.append('/arch:SSE2')

    openmp_enabled, needs_gomp = detect_openmp()
    if openmp_enabled:
        compiler_args.append('-fopenmp')
    compiler_libraries = ['gomp'] if needs_gomp else []

    dist = Extension('msmbuilder._distance_wrap', sources=glob('src/ext/scipy_distance/*.c'),
                     extra_compile_args=compiler_args,
                     libraries=compiler_libraries,
                     include_dirs=[numpy.get_include()])
    contact = Extension('msmbuilder._contact_wrap', sources=glob('src/ext/contact/*.c'),
                        extra_compile_args=compiler_args,
                        libraries=compiler_libraries,
                        include_dirs = [numpy.get_include()])        

    write_version_py()
    setup(ext_modules=[dist, contact], **metadata)

if __name__ == '__main__':
    if os.environ.get('READTHEDOCS', None) == 'True' and __name__ == '__main__':
        travis_main()
    else:
        main()
