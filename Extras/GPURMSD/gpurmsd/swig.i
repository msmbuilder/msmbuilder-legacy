%module swGPURMSD
%{
#define SWIG_FILE_WITH_INIT
#include "RMSD.hh"    
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (int DIM1, int DIM2, int DIM3, float* INPLACE_ARRAY3) {(int numAtoms, int numDimens, int numConfs, float* h_X)}
%apply (int DIM1, float* INPLACE_ARRAY1) {(int numConfs, float* h_rmsds)}
%apply (int DIM1, int* INPLACE_ARRAY1) {(int numConfs, int* h_subset_flag)}
%include "RMSD.hh"
