u"""
setup.py: Install msmbuilder.  
"""
import os, sys
from glob import glob
import subprocess
import shutil
import tempfile
from distutils.ccompiler import new_compiler

VERSION = "2.7.dev"
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
    'platforms': ["Linux", "Mac OS X"],
    'zip_safe': False,
    'description': "Python Code for Building Markov State Models",
    'long_description': """MSMBuilder (https://simtk.org/home/msmbuilder)
is a library that provides tools for analyzing molecular dynamics
simulations, particularly through the construction
of Markov state models for conformational dynamics."""}

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
    print 'Attempting to autodetect OpenMP support...', 
    hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
    needs_gomp = hasopenmp
    if not hasopenmp:
        compiler.add_library('gomp')
        hasopenmp = hasfunction(compiler, 'omp_get_num_threads()')
        needs_gomp = hasopenmp
    print
    if hasopenmp:
        print 'Compiler supports OpenMP'
    else:
        print 'Did not detect OpenMP support; parallel RMSD disabled'
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
                            'msmbuilder.geometry', 'msmbuilder.metrics', 'msmbuilder.reduce']
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

    config.add_subpackage('reduce',
                          subpackage_path='src/python/reduce')

    openmp_enabled, needs_gomp = detect_openmp()
    compiler_args = ['-O3', '-funroll-loops']
    if new_compiler().compiler_type == 'msvc':
        compiler_args.append('/arch:SSE2')

    if openmp_enabled:
        compiler_args.append('-fopenmp')
    compiler_libraries = ['gomp'] if needs_gomp else []
    #compiler_defs = [('USE_OPENMP', None)] if openmp_enabled else []


    # add asa extension
    # note this is wrapped using f2py, which
	# is a little different that the other modules,
    # but actually a lot more convenient and less error prone
    asa = Extension('msmbuilder._asa',
                    sources=['src/ext/asa/asa.pyf', 'src/ext/asa/asa.c'],
                    libraries=compiler_libraries,
                    extra_compile_args=compiler_args)

    
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
                                           "-msse2","-msse3"] + compiler_args,
                     libraries=compiler_libraries,
                     include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])

    for e in [asa, xtc, dcd, rmsd]:
        config.ext_modules.append(e)
        
    # add all of the distance metrics with the same compile_args, link_args, etc
    dist = Extension('msmbuilder._distance_wrap', sources=glob('src/ext/scipy_distance/*.c'))
    dihedral = Extension('msmbuilder._dihedral_wrap', sources=glob('src/ext/dihedral/*.c'))
    contact = Extension('msmbuilder._contact_wrap', sources=glob('src/ext/contact/*.c'))
    rg = Extension('msmbuilder._rg_wrap', sources=glob('src/ext/rg/*.c'))

    for ext in [dist, dihedral, contact, rg]:
        ext.extra_compile_args = compiler_args
        ext.extra_link_args = compiler_libraries
        ext.include_dirs = [numpy.get_include()]
        config.ext_modules.append(ext)
    
    return config

if __name__ == '__main__':
    write_version_py()
    metadata['configuration'] = configuration
    setup(**metadata)
