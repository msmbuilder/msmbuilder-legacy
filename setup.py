u"""
setup.py: Install msmbuilder.  
"""
VERSION="2.6.dev"
__author__ = "Kyle A. Beauchamp"
__version__ = VERSION

import os, sys, re
import platform
from distutils.core import setup,Extension
import distutils.sysconfig
import numpy
import glob

requirements = ['scipy', 'numpy', 'matplotlib', 'deap', 'fastcluster']

# Add something to the link line for OSX10.7
base_link_args = []
if platform.mac_ver()[0] != '': # if we're on a mac
    if int(platform.mac_ver()[0].split('.')[1]) >= 7: # if the OS X version number is greater than or equal to 7
        base_link_args.extend(['-L/Developer/SDKs/MacOSX10.6.sdk/usr/lib/'])

# If you're using EPD python distribution, which we recommend, the lib directory
# of your installation contains Intel MKL shared object files
epd_mkl = ['-L%s' % distutils.sysconfig.get_config_var('LIBDIR'),]

# Declare the C extension modules
XTC = Extension('msmbuilder/libxdrfile',
                sources = [
                    "src/ext/xdrfile-1.1b/src/trr2xtc.c",
                    "src/ext/xdrfile-1.1b/src/xdrfile.c",
                    "src/ext/xdrfile-1.1b/src/xdrfile_trr.c",
                    "src/ext/xdrfile-1.1b/src/xdrfile_xtc.c",
                            ],
                 extra_compile_args=[],
                 extra_link_args = base_link_args,
                 include_dirs = ["src/ext/xdrfile-1.1b/include/"]
                )
DCD = Extension('msmbuilder/dcdplugin_s',
                sources = [ "src/ext/molfile_plugin/dcdplugin_s.c" ],
                libraries=['m'],
                extra_link_args=base_link_args,
                include_dirs = ["src/ext/molfile_plugin/include/",
                                "src/ext/molfile_plugin"]
                )
IRMSD = Extension('msmbuilder/_rmsdcalc',
                  sources = ["src/ext/IRMSD/theobald_rmsd.c",
                             "src/ext/IRMSD/rmsd_numpy_array.c"],
                  extra_compile_args = ["-std=c99","-O2","-shared","-msse2","-msse3","-fopenmp"],
                  extra_link_args = base_link_args + ['-lgomp'],
                  include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')]
                  )
LPRMSD = Extension('msmbuilder/_lprmsd',
                   sources = ["src/ext/LPRMSD/apc.c",
                              "src/ext/LPRMSD/qcprot.c",
                              "src/ext/LPRMSD/theobald_rmsd.c",
                              "src/ext/LPRMSD/lprmsd.c"],
                   include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')],
                   extra_compile_args=["-std=c99","-O2","-shared","-msse2","-msse3","-Wno-unused","-fopenmp","-m64"],
                   ##
                   # Note: LPRMSD is hard to compile because it uses the C interface to BLAS, which isn't installed on all operating systems.
                   ##
                   # Intel 11.1 MKL link line - it should work on any machine with the Intel compiler installed.
                   # Make sure that the directories containing the MKL libraries are correct.
                   #extra_link_args=['/opt/intel/Compiler/11.1/072/mkl/lib/em64t/libmkl_solver_lp64_sequential.a',
                   #                 '-Wl,--start-group','/opt/intel/Compiler/11.1/072/mkl/lib/em64t/libmkl_intel_lp64.a',
                   #                 '/opt/intel/Compiler/11.1/072/mkl/lib/em64t/libmkl_sequential.a',
                   #                 '/opt/intel/Compiler/11.1/072/mkl/lib/em64t/libmkl_core.a',
                   #                 '-Wl,--end-group','-lpthread','-lm','-lgomp']
                   
                   extra_link_args = base_link_args + epd_mkl + \
                        ['-latlas','-lcblas'] + ['-lpthread', '-lm', '-lgomp'],

                   )
DISTANCE = Extension('msmbuilder/_distance_wrap',
                      sources = ["src/ext/scipy_distance/distance.c",
                                 "src/ext/scipy_distance/distance_wrap.c"],
                      extra_compile_args = ["-std=c99","-O3","-shared","-msse2",
                                          "-msse3","-fopenmp", "-Wall"],
                      extra_link_args = base_link_args + ['-lgomp'],
                      include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])
DIHEDRAL = Extension('msmbuilder/_dihedral_wrap',
                     sources = ["src/ext/dihedral/dihedral.c",
                                "src/ext/dihedral/dihedral_wrap.c"],
                     extra_compile_args = ["-std=c99","-O3","-shared",
                                         "-fopenmp", "-Wall"],
                     extra_link_args = base_link_args + ['-lgomp'],
                     include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])
CONTACT = Extension('msmbuilder/_contact_wrap',
                    sources = ["src/ext/contact/contact.c",
                               "src/ext/contact/contact_wrap.c"],
                    extra_compile_args=["-std=c99","-O3","-shared",
                                        "-fopenmp", "-Wall"],
                    extra_link_args = base_link_args + ['-lgomp'],
                    include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])
RG = Extension('msmbuilder/_rg_wrap',
               sources = ["src/ext/rg/rg.c",
                          "src/ext/rg/rg_wrap.c"],
               extra_compile_args=["-std=c99","-O3","-shared",
                                   "-fopenmp", "-Wall"],
               extra_link_args = base_link_args + ['-lgomp'],
               include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])


def buildKeywordDictionary(use_LPRMSD=True):
    from distutils.core import Extension
    setupKeywords = {}
    setupKeywords["name"]              = "msmbuilder"
    setupKeywords["version"]           = VERSION
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
    setupKeywords["scripts"]           = filter(lambda elem: not re.search( "^__", elem ), glob.glob('scripts/*'))
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
         )]
    
    setupKeywords["ext_modules"] = [IRMSD, XTC, DCD, DISTANCE, DIHEDRAL, CONTACT, RG]
    if use_LPRMSD:
        setupKeywords['ext_modules'].append(LPRMSD)
    

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
    try:
        setup_keywords = buildKeywordDictionary()
        setup(**setup_keywords)
    except:
        BUILD_EXT_WARNING = "WARNING: The C extension 'LPRMSD' could not be compiled.\nThis may be due to a failure to find the BLAS/MKL libraries." 
        print '*' * 75
        print BUILD_EXT_WARNING
        print "I'm retrying the build without the C extension now."
        print '*' * 75

        setup_keywords = buildKeywordDictionary(use_LPRMSD=False)
        setup(**setup_keywords)

        print '*' * 75
        print BUILD_EXT_WARNING
        print '*' * 75
        

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

