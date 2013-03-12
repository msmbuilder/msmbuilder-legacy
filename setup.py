u"""
setup.py: Install msmbuilder.  
"""
import os, sys
from glob import glob
import subprocess

VERSION = "2.6.dev"
ISRELEASED = False
__author__ = "MSMBuilder Team"
__version__ = VERSION

# metadata for setup()
metadata = {
    'version': VERSION,
    'author': __author__,
    'author_email': 'kyleb@stanford.edu',
    'license': 'GPL v3.0',
    'url': 'https://simtk.org/home/msmbuilder',
    'download_url': 'https://simtk.org/home/msmbuilder',
    'install_requires': ['scipy', 'matplotlib', 'pyyaml',
                         'deap', 'fastcluster','statsmodels',
                         'pandas', 'tables'],
    'platforms': ["Linux", "Mac OS X"],
    'zip_safe': False,
    'description': "Python Code for Building Markov State Models",
    'long_description': """MSMBuilder (https://simtk.org/home/msmbuilder)
is a library that provides tools for analyzing molecular dynamics
simulations, particularly through the construction
of Markov state models for conformational dynamics."""}


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


if os.environ.get('READTHEDOCS', None) == 'True' and __name__ == '__main__':
    # On READTHEDOCS, the service that hosts our documentation, the build
    # environment does not have numpy and cannot build C extension modules,
    # so if we detect this environment variable, we're going to bail out
    # and run a minimal setup. This only installs the python packages, which
    # is not enough to RUN anything, but should be enough to introspect the
    # docstrings, which is what's needed for the documentation
    from distutils.core import setup
    import tempfile, shutil
    write_version_py()
    
    metadata['name'] = 'msmbuilder'
    metadata['packages'] = ['msmbuilder', 'msmbuilder.scripts', 'msmbuilder.project',
                            'msmbuilder.geometry', 'msmbuilder.metrics']
    metadata['scripts'] = [e for e in glob('scripts/*.py') if not e.endswith('__.py')]

    # dirty, dirty trick to install "mock" packages
    mockdir = tempfile.mkdtemp()
    open(os.path.join(mockdir, '__init__.py'), 'w').close()
    extensions = ['msmbuilder._distance_wrap', 'msmbuilder._rmsdcalc',
                   'msmbuilder._asa', 'msmbuilder._rg_wrap',
                   'msmbuilder._distance_wrap', 'msmbuilder._contact_wrap',
                   'msmbuilder._dihedral_wrap']
    metadata['package_dir'] = {'msmbuilder': 'src/python', 'msmbuilder.scripts': 'scripts'}
    metadata['packages'].extend(extensions)
    for ex in extensions:
        metadata['package_dir'][ex] = mockdir
    # end dirty trick :)

    setup(**metadata)
    shutil.rmtree(mockdir) #clean up dirty trick
    sys.exit(1)


# now procede to standard setup
# setuptools needs to come before numpy.distutils to get install_requires
import setuptools 
import numpy
from distutils import sysconfig
from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration

def configuration(parent_package='',top_path=None):
    "Configure the build"

    config = Configuration('msmbuilder',
                           package_parent=parent_package,
                           top_path=top_path,
                           package_path='src/python')
    config.set_options(assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=False)
    
    #once all of the data is in one place, we can add it with this
    config.add_data_dir('reference')
    
    # add the scipts, so they can be called from the command line
    config.add_scripts([e for e in glob('scripts/*.py') if not e.endswith('__.py')])
    
    # add scripts as a subpackage (so they can be imported from other scripts)
    config.add_subpackage('scripts',
                          subpackage_path=None)

    # add geometry subpackage
    config.add_subpackage('geometry',
                          subpackage_path='src/python/geometry')

    config.add_subpackage('project',
                          subpackage_path='src/python/project')

    config.add_subpackage('lumping',
                          subpackage_path='src/python/lumping')

    # add asa extension
    # note this is wrapped using f2py, which
	# is a little different that the other modules,
    # but actually a lot more convenient and less error prone
    asa = Extension('msmbuilder._asa',
                    sources = ['src/ext/asa/asa.pyf', 'src/ext/asa/asa.c'],
                    extra_link_args = ['-lgomp'],
                    extra_compile_args = ['-fopenmp'])

    
    # add metrics subpackage
    config.add_subpackage('metrics',
                          subpackage_path='src/python/metrics')

    #xtc reader
    xtc = Extension('msmbuilder.libxdrfile',
                    sources = ['src/ext/xdrfile-1.1b/src/xdrfile.c',
                               'src/ext/xdrfile-1.1b/src/trr2xtc.c',
                               'src/ext/xdrfile-1.1b/src/xdrfile_trr.c',
                               'src/ext/xdrfile-1.1b/src/xdrfile_xtc.c'],
                    include_dirs = ["src/ext/xdrfile-1.1b/include/"])
    # dcd reader
    dcd = Extension('msmbuilder.dcdplugin_s',
                    sources = ["src/ext/molfile_plugin/dcdplugin_s.c"],
                    libraries=['m'],
                    include_dirs = ["src/ext/molfile_plugin/include/",
                                    "src/ext/molfile_plugin"])
    # rmsd
    rmsd = Extension('msmbuilder._rmsdcalc',
                     sources=glob('src/ext/IRMSD/*.c'),
                     extra_compile_args = ["-std=c99","-O2",
                                           "-msse2","-msse3","-fopenmp"],
                     extra_link_args = ['-lgomp'],
                     include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])

    for e in [asa, xtc, dcd, rmsd]:
        config.ext_modules.append(e)
        
    # add all of the distance metrics with the same compile_args, link_args, etc
    dist = Extension('msmbuilder._distance_wrap', sources=glob('src/ext/scipy_distance/*.c'))
    dihedral = Extension('msmbuilder._dihedral_wrap', sources=glob('src/ext/dihedral/*.c'))
    contact = Extension('msmbuilder._contact_wrap', sources=glob('src/ext/contact/*.c'))
    rg = Extension('msmbuilder._rg_wrap', sources=glob('src/ext/rg/*.c'))

    for ext in [dist, dihedral, contact, rg]:
        ext.extra_compile_args = ["-O3", "-fopenmp", "-Wall"]
        ext.extra_link_args = ['-lgomp']
        ext.include_dirs = [numpy.get_include()]
        config.ext_modules.append(ext)
    
    return config

if __name__ == '__main__':
    write_version_py()
    metadata['configuration'] = configuration
    setup(**metadata)
