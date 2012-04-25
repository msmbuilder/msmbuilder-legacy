u"""
setup.py: Install msmbuilder.  
"""
VERSION=205
__author__ = "Kyle A. Beauchamp"
__version__ = "%d"%VERSION

import os, sys
from distutils.core import setup,Extension
import numpy
import glob

requirements = ['scipy', 'numpy', 'matplotlib', 'deap', 'fastcluster']

# Declare the C extension modules
XTC = Extension('msmbuilder/libxdrfile',
                  sources = [
                    "src/ext/xdrfile-1.1b/src/trr2xtc.c",
                    "src/ext/xdrfile-1.1b/src/xdrfile.c",
                    "src/ext/xdrfile-1.1b/src/xdrfile_trr.c",
                    "src/ext/xdrfile-1.1b/src/xdrfile_xtc.c",
                            ],
                  extra_compile_args=[],
                  extra_link_args=["--enable-shared"],
                  include_dirs = ["src/ext/xdrfile-1.1b/include/"]
                  )
DCD = Extension('msmbuilder/dcdplugin_s',
                sources = [ "src/ext/molfile_plugin/dcdplugin_s.c" ],
                libraries=['m'],
                include_dirs = ["src/ext/molfile_plugin/include/",
                                "src/ext/molfile_plugin"]
                )
IRMSD = Extension('msmbuilder/_rmsdcalc',
                  sources = ["src/ext/IRMSD/theobald_rmsd.c",
                             "src/ext/IRMSD/rmsd_numpy_array.c"],
                  extra_compile_args=["-std=c99","-O2","-shared","-msse2","-msse3","-fopenmp"],
                  extra_link_args=['-lgomp'],
                  include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')]
                  )
LPRMSD = Extension('msmbuilder/_lprmsd',
                   sources = ["src/ext/LPRMSD/apc.c",
                              "src/ext/LPRMSD/qcprot.c",
                              "src/ext/LPRMSD/theobald_rmsd.c",
                              "src/ext/LPRMSD/lprmsd.c"],
                   extra_compile_args=["-std=c99","-O2","-shared","-msse2","-msse3","-fopenmp","-Wno-unused","-m64","-I/opt/intel/Compiler/11.1/072/mkl/include"],
                   extra_link_args=['-lgomp','-lblas'],
                   #Uncomment below to use Intel BLAS
                   #extra_link_args=['/opt/intel/Compiler/11.1/072/mkl/lib/em64t/libmkl_solver_lp64_sequential.a',
                   #                 '-Wl,--start-group','/opt/intel/Compiler/11.1/072/mkl/lib/em64t/libmkl_intel_lp64.a',
                   #                 '/opt/intel/Compiler/11.1/072/mkl/lib/em64t/libmkl_sequential.a',
                   #                 '/opt/intel/Compiler/11.1/072/mkl/lib/em64t/libmkl_core.a',
                   #                 '-Wl,--end-group','-lpthread','-lm','-lgomp'],


                   include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')]
                   )
DISTANCE = Extension('msmbuilder/_distance_wrap',
                      sources = ["src/ext/scipy_distance/distance.c",
                                 "src/ext/scipy_distance/distance_wrap.c"],
                      extra_compile_args=["-std=c99","-O3","-shared","-msse2",
                                          "-msse3","-fopenmp", "-Wall"],
                      extra_link_args=['-lgomp'],
                      include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])
DIHEDRAL = Extension('msmbuilder/_dihedral_wrap',
                     sources = ["src/ext/dihedral/dihedral.c",
                                "src/ext/dihedral/dihedral_wrap.c"],
                     extra_compile_args=["-std=c99","-O3","-shared",
                                         "-fopenmp", "-Wall"],
                     extra_link_args=['-lgomp'],
                     include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])
CONTACT = Extension('msmbuilder/_contact_wrap',
                    sources = ["src/ext/contact/contact.c",
                               "src/ext/contact/contact_wrap.c"],
                    extra_compile_args=["-std=c99","-O3","-shared",
                                        "-fopenmp", "-Wall"],
                    extra_link_args=['-lgomp'],
                    include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])
RG = Extension('msmbuilder/_rg_wrap',
               sources = ["src/ext/rg/rg.c",
                          "src/ext/rg/rg_wrap.c"],
               extra_compile_args=["-std=c99","-O3","-shared",
                                   "-fopenmp", "-Wall"],
               extra_link_args=['-lgomp'],
               include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])


def buildKeywordDictionary():
    from distutils.core import Extension
    setupKeywords = {}
    setupKeywords["name"]              = "msmbuilder"
    setupKeywords["version"]           = "%d" %VERSION
    setupKeywords["author"]            = "Kyle A. Beauchamp"
    setupKeywords["author_email"]      = "kyleb@stanford.edu"
    setupKeywords["license"]           = "GPL 3.0"
    setupKeywords["url"]               = "https://simtk.org/home/msmbuilder"
    setupKeywords["download_url"]      = "https://simtk.org/home/msmbuilder"
    setupKeywords["packages"]          = ["msmbuilder",
                                          "msmbuilder.scripts",
                                          "msmbuilder.geometry"]
    setupKeywords["package_dir"]       = {"msmbuilder": "src/python",
                                          "msmbuilder.scripts": "scripts/"}
    setupKeywords["scripts"]           = glob.glob("scripts/*.py")
    setupKeywords["package_data"]      = {
        "msmbuilder"                   : ["AUTHORS","COPYING"]
                                         }
    setupKeywords["data_files"]        = [
        (
        "share/msmbuilder_tutorial/",[
        "Tutorial/AtomIndices.dat",
        "Tutorial/native.pdb",
        "Tutorial/Phi.h5",
        "Tutorial/Psi.h5",
        "Tutorial/PlotDihedrals.py",
        "Tutorial/PlotMacrostateImpliedTimescales.py"
        ]),
        (
        "share/msmbuilder_tutorial/",["Tutorial/XTC.tar"]
         )
                                          ]
    setupKeywords["ext_modules"]       = [IRMSD, LPRMSD, XTC, DCD, DISTANCE, DIHEDRAL, CONTACT, RG]
    setupKeywords["platforms"]         = ["Linux", "Mac OS X", "Windows"]
    setupKeywords["description"]       = "Python Code for Building Markov State Models."
    setupKeywords["long_description"]  = """
    MSMBuilder (https://simtk.org/home/msmbuilder) is a library
    that provides tools for analyzing molecular dynamics simulations, particularly through the construction
    of Markov state models for conformational dynamics.  
    """
    outputString=""
    firstTab     = 40
    secondTab    = 60
    for key in sorted( setupKeywords.iterkeys() ):
         value         = setupKeywords[key]
         outputString += key.rjust(firstTab) + str( value ).rjust(secondTab) + "\n"
    
    print "%s" % outputString

    return setupKeywords
    

def main():
    setupKeywords=buildKeywordDictionary()
    setup(**setupKeywords)
    for requirement in requirements:
      try:
          exec('import %s' % requirement)
      except ImportError as e:
          print >> sys.stderr, '\nWarning: Could not import %s' % e
          print >> sys.stderr, 'Warning: Some package functionality many not work'
          if requirement == 'deap':
              print >> sys.stderr, "\nthe 'deap' package contains python tools for MPI"
              print >> sys.stderr, 'it can be installed with "easy_install deap"'
              print >> sys.stderr, 'it is not required.'
          if requirement == 'fastcluster':
              print >> sys.stderr, "\nthe 'fastcluster' package contains fast implementations"
              print >> sys.stderr, 'of hierarchical clustering algorithms.'
              print >> sys.stderr, 'it can be downloaded from http://cran.r-project.org/web/packages/fastcluster/'
              print >> sys.stderr, '(get the download called "package source")'
              print >> sys.stderr, 'it is not required.'

if __name__ == '__main__':
    main()

