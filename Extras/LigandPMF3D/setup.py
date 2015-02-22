import sys
from distutils.version import StrictVersion
import glob
from setuptools import setup

def warn_on_version(module_name, minimum=None, package_name=None, recommend_conda=True):
    if package_name is None:
        package_name = module_name

    class VersionError(Exception):
        pass

    msg = None
    try:
        package = __import__(module_name)
        if minimum is not None:
            try:
                v = package.version.short_version
            except AttributeError:
                v = package.__version__
            if StrictVersion(v) < StrictVersion(minimum):
                raise VersionError
    except ImportError:
        if minimum is None:
            msg = 'LigandPMF3D requires the python package "%s", which is not installed.' % package_name
        else:
            msg = 'LigandPMF3D requires the python package "%s", version %s or later.' % (package_name, minimum)
    except VersionError:    
        msg = ('LigandPMF3D requires the python package "%s", version %s or '
               ' later. You have version %s installed. You will need to upgrade.') % (package_name, minimum, v)

    if recommend_conda:
        install = ('\nTo install %s, we recommend the conda package manger. See http://conda.pydata.org for info on conda.\n'
                   'Using conda, you can install it with::\n\n    $ conda install %s') % (package_name, package_name)
        install += '\n\nAlternatively, with pip you can install the package with:\n\n    $ pip install %s' % package_name
    else:
        install = '\nWith pip you can install the package with:\n\n    $ pip install %s' % package_name
    
    if msg:
        banner = ('==' * 40)
        print('\n'.join([banner, banner, "", msg, install, "", banner, banner]))



setup(name='PMF3D',
      author='Morgan Lawrenz', 
      author_email='morganlawrenz@gmail.com',
      version = '1.0',
      description = 'Protein-ligand binding 3-D free energy maps', 
      py_modules = ['PMF3D'],
      scripts=glob.glob('RunPMF.py'), )
warn_on_version('mdtraj', '0.8.0')

