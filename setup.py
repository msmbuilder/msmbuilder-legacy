u"""
setup.py: Install msmbuilder.  
"""

VERSION="2.6.dev"
__author__ = "MSMBuilder Team"
__version__ = VERSION

import os, sys
from copy import deepcopy
from glob import glob
# setuptools needs to come before numpy.distutils to get install_requires
import setuptools 
import numpy
from distutils import sysconfig
from numpy.distutils.core import setup, Extension
from numpy.distutils.misc_util import Configuration

# metadata for setup()
metadata = {
    'version': VERSION,
    'author': __author__,
    'author_email': 'kyleb@stanford.edu',
    'license': 'GPL v3.0',
    'url': 'https://simtk.org/home/msmbuilder',
    'download_url': 'https://simtk.org/home/msmbuilder',
    'install_requires': ['scipy', 'numpy', 'matplotlib',
                         'deap', 'fastcluster'],
    'platforms': ["Linux", "Mac OS X"],
    'description': "Python Code for Building Markov State Models",
    'long_description': """MSMBuilder (https://simtk.org/home/msmbuilder)
is a library that provides tools for analyzing molecular dynamics
simulations, particularly through the construction                                                    
of Markov state models for conformational dynamics."""}

def configuration(parent_package='',top_path=None):
    "Configure the build"

    config = Configuration('msmbuilder',
                           package_parent=parent_package,
                           top_path=top_path,
                           package_path='src/python')
    config.set_options(assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    # add the scipts, so they can be called from the command line
    config.add_scripts([e for e in glob('scripts/*.py') if not e.endswith('__.py')])
    
    # add scripts as a subpackage (so they can be imported from other scripts)
    config.add_subpackage('scripts',
                          subpackage_path=None)

    # add geometry subpackage
    config.add_subpackage('geometry',
                          subpackage_path='src/python/geometry')
    
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
                     include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])
    
    for e in [xtc, dcd, rmsd]:
        config.ext_modules.append(e)
        
    # add all of the distance metrics with the same compile_args, link_args, etc
    dist = Extension('msmbuilder._distance_wrap', sources=glob('src/ext/scipy_distance/*.c'))
    dihedral = Extension('msmbuilder._dihedral_wrap', sources=glob('src/ext/dihedral/*.c'))
    contact = Extension('msmbuilder._contact_wrap', sources=glob('src/ext/contact/*.c'))
    rg = Extension('msmbuilder._rg_wrap', sources=glob('src/ext/dihedral/*.c'))

    for ext in [dist, dihedral, contact, rg]:
        ext.extra_compile_args = ["-O3", "-fopenmp", "-Wall"]
        ext.extra_link_args = ['-lgomp']
        ext.include_dirs = [numpy.get_include()]
        config.ext_modules.append(ext)
    
    return config

def add_lprmsd_and_setup():
    """LPRMSD depends on the C interface to BLAS, which is hard to find. This method
    trys building it by looking in a few places sequentially."""

    # the basic extension
    lprmsd = Extension('msmbuilder._lprmsd',
                       sources = glob('src/ext/LPRMSD/*.c'),
                       extra_compile_args = ["-std=c99","-O2",
                                             "-msse2","-msse3","-Wno-unused","-fopenmp","-m64"],
                       extra_link_args = ['-lpthread', '-lm', '-lgomp'],
                       include_dirs = [numpy.get_include(),
                                       os.path.join(numpy.get_include(), 'numpy')])
    
    # the extra link lines we're going to try
    attempts = [{'extra_link_args' : []},
                {'extra_link_args': ['-latlas']},
                {'extra_link_args': ['-lcblas']}]

    failure = True

    for group in attempts:
        e = deepcopy(lprmsd)
        if 'extra_link_args' in group:
            e.extra_link_args.extend(group['extra_link_args'])
        
        try:
            mymeta = deepcopy(metadata)
            mymeta['ext_modules'] = [e]
            mymeta['configuration'] = configuration
            setup(**mymeta)
            failure = False
            break

        except SystemExit as e:
            pass

    if failure:
        print '\n' + '#'*70 + '\n'
        print 'I could not compile the LPRMSD distance metric, which depends on the C'
        print 'inteface to BLAS -- the basic linear algebra subroutines. (The fortran'
        print 'interface is the more common code.'
        print 'I tried using both "-latlas", and "-lcblas", along with not giving any'
        print 'extra link line arguments, and only including the python libdir'
        print '(%s)' %sysconfig.get_config_var('LIBDIR')
        print 'which, on the Enthought python distribution (academic version), includes'
        print 'Intel MKL shared object files'
        print '\n' + '#'*70 + '\n'
        

if __name__ == '__main__':
    add_lprmsd_and_setup()
