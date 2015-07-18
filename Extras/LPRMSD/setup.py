from __future__ import print_function
import os, sys
import numpy
from glob import glob
import tempfile
import shutil
from setuptools import setup, Extension
from distutils.ccompiler import new_compiler

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
    print('Attempting to autodetect OpenMP support...', end=' ')
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


hasopenmp, needs_gomp = detect_openmp()

# If you are 32-bit you should remove the -m64 flag
compile_args = ["-std=c99", "-O2", "-msse2", "-msse3", 
                "-Wno-unused", "-m64"]

extra_link_args = ["-lblas", "-lpthread", "-lm"]

if hasopenmp:
    compile_args.append("-fopenmp")

if needs_gomp:
    extra_link_args.append("-gomp")

_lprmsd = Extension('_lprmsd',
          sources = glob('src/*.c'),
          extra_compile_args = compile_args,
          extra_link_args = extra_link_args,
          include_dirs = [numpy.get_include(), os.path.join(numpy.get_include(), 'numpy')])

setup(name='msmbuilder.metrics.lprmsd',
      version='1.2',
      py_modules = ['lprmsd'],
      ext_modules = [_lprmsd],
      scripts=glob('scripts/*.py'),
      zip_safe=False,
      entry_points="""
        [msmbuilder.metrics]
         metric_class=lprmsd:LPRMSD
         add_metric_parser=lprmsd:add_metric_parser
         construct_metric=lprmsd:construct_metric
         """)
