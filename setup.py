"""
setup.py: Install msmbuilder.  
"""
VERSION=200
__author__ = "Kyle A. Beauchamp"
__version__ = "%d"%VERSION

from distutils.core import setup,Extension
import numpy
import glob

IRMSD = Extension('msmbuilder/rmsdcalc',
                  sources = ["msmbuilder/IRMSD/theobald_rmsd.c","msmbuilder/IRMSD/rmsd_numpy_array.c"],
                  extra_compile_args=["-std=c99","-O2","-shared","-msse2","-msse3","-fopenmp"],
                  extra_link_args=['-lgomp'],
                  include_dirs = [numpy.get_include(),numpy.get_include()+"/numpy/"]
                  )
XTC = Extension('msmbuilder/libxdrfile',
                  sources = [
                    "xdrfile-1.1b/src/trr2xtc.c",
                    "xdrfile-1.1b/src/xdrfile.c",
                    "xdrfile-1.1b/src/xdrfile_trr.c",
                    "xdrfile-1.1b/src/xdrfile_xtc.c",
                            ],
                  extra_compile_args=[],
                  extra_link_args=["--enable-shared"],
                  include_dirs = ["xdrfile-1.1b/include/"]
                  )
DCD = Extension('msmbuilder/dcdplugin_s',
                sources = [ "molfile_plugin/dcdplugin_s.c" ],
                libraries=['m'],
                include_dirs = ["molfile_plugin/include/","molfile_plugin"] 
                )

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
    setupKeywords["packages"]          = ["msmbuilder","msmbuilder/Scripts/"]
    setupKeywords["scripts"]           = glob.glob("msmbuilder/Scripts/*.py")
    setupKeywords["package_data"]      = {
        "msmbuilder"                   : ["AUTHORS","COPYING"]
                                         }
    setupKeywords["data_files"]        = []
    setupKeywords["ext_modules"]       = [IRMSD,XTC,DCD]
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

if __name__ == '__main__':
    main()




