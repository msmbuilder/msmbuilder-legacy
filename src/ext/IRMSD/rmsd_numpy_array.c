// This file is part of MSMBuilder.
//
// Copyright 2011 Stanford University
//
// MSMBuilder is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//


#include "Python.h"
#include "arrayobject.h"
#include <stdint.h>
#include <stdio.h>
#include "theobald_rmsd.h"
#include <omp.h>


#define CHECKARRAYTYPE(ary,name) if (PyArray_TYPE(ary) != NPY_FLOAT32) {\
                                     PyErr_SetString(PyExc_ValueError,name" was not of type float32");\
                                     return NULL;\
                                 } 
#define CHECKARRAYCARRAY(ary,name) if ((PyArray_FLAGS(ary) & NPY_CARRAY) != NPY_CARRAY) {\
                                       PyErr_SetString(PyExc_ValueError,name" was not a contiguous well-behaved array in C order");\
                                       return NULL;\
                                   } 


static PyObject *_getMultipleRMSDs_aligned_T_g(PyObject *self, PyObject *args) {
    npy_intp dim2[2];
    float *AData,*BData,*GAData;
    npy_intp *arrayADims,*arrayBDims,*arrayAStrides,*arrayBStrides,*tan_strides;
    int nrefmols,nqmols;
    PyArrayObject* ArrayDistances;
    float* Distances;
    float *ADataet,*refcountset;
    float refmag,reflength;
    float* outputrow;
    int row,col;
    float t;
    int nprocs=1;
    float rmsd2;
    float G_x=-1,G_y=-1;
    int nrealatoms=-1,npaddedatoms=-1,rowstride=-1;
    int truestride=-1;
    PyArrayObject *ary_coorda, *ary_coordb,*ary_Ga;
 
    /*ultimately OO|ff with optimal G's*/
 
    if (!PyArg_ParseTuple(args, "iiiOOOf",&nrealatoms,&npaddedatoms,&rowstride,
              &ary_coorda, &ary_coordb,&ary_Ga,&G_y)) {
      return NULL;
    }

  
    // Get pointers to array data
    AData  = (float*) PyArray_DATA(ary_coorda);
    BData  = (float*) PyArray_DATA(ary_coordb);
    GAData  = (float*) PyArray_DATA(ary_Ga);

    // TODO add sanity checking on Ga

    // Get dimensions of arrays (# molecules, maxlingos)
    // Note: strides are in BYTES, not INTs
    arrayADims = PyArray_DIMS(ary_coorda);
    arrayAStrides = PyArray_STRIDES(ary_coorda);
    arrayBDims = PyArray_DIMS(ary_coordb);
    arrayBStrides = PyArray_STRIDES(ary_coordb);

    // Do some sanity checking on array dimensions
    //      - make sure they are of float32 data type
    CHECKARRAYTYPE(ary_coorda,"Array A");
    CHECKARRAYTYPE(ary_coordb,"Array B");

    //      - make sure lingo/count/mag arrays are 2d and are the same size in a set (ref/q)
    if (ary_coorda->nd != 3) {
        PyErr_SetString(PyExc_ValueError,"Array A did not have dimension 3");
	return NULL;
    }
    if (ary_coordb->nd != 2) {
        PyErr_SetString(PyExc_ValueError,"Array B did not have dimension 2");
        return NULL;
    }
    //      - make sure stride is 4 in last dimension (ie, is C-style and contiguous)
    CHECKARRAYCARRAY(ary_coorda,"Array A");
    CHECKARRAYCARRAY(ary_coordb,"Array B");


    // Create return array containing Distances
    dim2[0] = arrayADims[0];
    dim2[1] = 1;
    ArrayDistances = (PyArrayObject*) PyArray_SimpleNew(1,dim2,NPY_FLOAT);
    Distances = (float*) PyArray_DATA(ArrayDistances);
    
    truestride=npaddedatoms*3;
    
    #pragma omp parallel for
    for (int i = 0; i < arrayADims[0]; i++)  {
        float msd = ls_rmsd2_aligned_T_g(nrealatoms,npaddedatoms,rowstride,(AData+i*truestride),BData,GAData[i],G_y);
        Distances[i] = sqrtf(msd);
    }
    
    return PyArray_Return(ArrayDistances);
}

static PyObject *_getMultipleRMSDs_aligned_T_g_at_indices(PyObject *self, PyObject *args) {
    npy_intp dim2[2];
    float *AData,*BData,*GAData;
    npy_intp *arrayADims,*arrayBDims,*arrayAStrides,*arrayBStrides,*tan_strides;
    int nrefmols,nqmols;
    PyArrayObject* ArrayDistances;
    float* Distances;
    float *ADataet,*refcountset;
    float refmag,reflength;
    float* outputrow;
    int row,col;
    float t;
    int nprocs=1;
    float rmsd2;
    float G_x=-1,G_y=-1;
    int nrealatoms=-1,npaddedatoms=-1,rowstride=-1;
    int truestride=-1;
    PyArrayObject *ary_coorda, *ary_coordb,*ary_Ga;
    PyArrayObject *ary_indices;
    npy_intp *Indices;
    npy_intp *arrayIndicesDims;

    /*ultimately OO|ff with optimal G's*/
 
    if (!PyArg_ParseTuple(args, "iiiOOOfO",&nrealatoms,&npaddedatoms,&rowstride,
            &ary_coorda, &ary_coordb,&ary_Ga,&G_y,&ary_indices)) {
      return NULL;
    }

  
    // Get pointers to array data
    AData  = (float*) PyArray_DATA(ary_coorda);
    BData  = (float*) PyArray_DATA(ary_coordb);
    GAData  = (float*) PyArray_DATA(ary_Ga);
    Indices = (npy_intp*) PyArray_DATA(ary_indices);

    // TODO add sanity checking on Ga

    // Get dimensions of arrays (# molecules, maxlingos)
    // Note: strides are in BYTES, not INTs
    arrayADims = PyArray_DIMS(ary_coorda);
    arrayAStrides = PyArray_STRIDES(ary_coorda);
    arrayBDims = PyArray_DIMS(ary_coordb);
    arrayBStrides = PyArray_STRIDES(ary_coordb);

    arrayIndicesDims = PyArray_DIMS(ary_indices);

    // Do some sanity checking on array dimensions
    //      - make sure they are of float32 data type
    CHECKARRAYTYPE(ary_coorda,"Array A");
    CHECKARRAYTYPE(ary_coordb,"Array B");

    //      - make sure lingo/count/mag arrays are 2d and are the same size in a set (ref/q)
    if (ary_coorda->nd != 3) {
        PyErr_SetString(PyExc_ValueError,"Array A did not have dimension 3");
        return NULL;
    }
    if (ary_coordb->nd != 2) {
        PyErr_SetString(PyExc_ValueError,"Array B did not have dimension 2");
        return NULL;
    }
    //      - make sure stride is 4 in last dimension (ie, is C-style and contiguous)
    CHECKARRAYCARRAY(ary_coorda,"Array A");
    CHECKARRAYCARRAY(ary_coordb,"Array B");


    // Create return array containing Distances
    dim2[0] = arrayIndicesDims[0];
    dim2[1] = 1;
    ArrayDistances = (PyArrayObject*) PyArray_SimpleNew(1,dim2,NPY_FLOAT);
    Distances = (float*) PyArray_DATA(ArrayDistances);
    
    truestride=npaddedatoms*3;
    
    #pragma omp parallel for
    for (int i = 0; i < arrayIndicesDims[0]; i++) {
        float msd = ls_rmsd2_aligned_T_g(nrealatoms, npaddedatoms, rowstride, AData+Indices[i]*truestride, BData, GAData[Indices[i]], G_y);
        Distances[i] = sqrtf(msd);
    }
    
    return PyArray_Return(ArrayDistances);
}

static PyObject *_getMultipleRMSDs_aligned_T_g_at_indices_serial(PyObject *self, PyObject *args) {
    npy_intp dim2[2];
    float *AData,*BData,*GAData;
    npy_intp *arrayADims,*arrayBDims,*arrayAStrides,*arrayBStrides,*tan_strides;
    int nrefmols,nqmols;
    PyArrayObject* ArrayDistances;
    float* Distances;
    float *ADataet,*refcountset;
    float refmag,reflength;
    float* outputrow;
    int row,col;
    float t;
    int nprocs=1;
    float rmsd2;
    float G_x=-1,G_y=-1;
    int nrealatoms=-1,npaddedatoms=-1,rowstride=-1;
    int truestride=-1;
    PyArrayObject *ary_coorda, *ary_coordb,*ary_Ga;
    PyArrayObject *ary_indices;
    npy_intp *Indices;
    npy_intp *arrayIndicesDims;

    /*ultimately OO|ff with optimal G's*/
 
    if (!PyArg_ParseTuple(args, "iiiOOOfO",&nrealatoms,&npaddedatoms,&rowstride,
			  &ary_coorda, &ary_coordb,&ary_Ga,&G_y,&ary_indices)) {
      return NULL;
    }

  
    // Get pointers to array data
    AData  = (float*) PyArray_DATA(ary_coorda);
    BData  = (float*) PyArray_DATA(ary_coordb);
    GAData  = (float*) PyArray_DATA(ary_Ga);
    Indices = (npy_intp*) PyArray_DATA(ary_indices);

    // TODO add sanity checking on Ga

    // Get dimensions of arrays (# molecules, maxlingos)
    // Note: strides are in BYTES, not INTs
    arrayADims = PyArray_DIMS(ary_coorda);
    arrayAStrides = PyArray_STRIDES(ary_coorda);
    arrayBDims = PyArray_DIMS(ary_coordb);
    arrayBStrides = PyArray_STRIDES(ary_coordb);

    arrayIndicesDims = PyArray_DIMS(ary_indices);

    // Do some sanity checking on array dimensions
    //      - make sure they are of float32 data type
    CHECKARRAYTYPE(ary_coorda,"Array A");
    CHECKARRAYTYPE(ary_coordb,"Array B");

    //      - make sure lingo/count/mag arrays are 2d and are the same size in a set (ref/q)
    if (ary_coorda->nd != 3) {
        PyErr_SetString(PyExc_ValueError,"Array A did not have dimension 3");
	return NULL;
    }
    if (ary_coordb->nd != 2) {
        PyErr_SetString(PyExc_ValueError,"Array B did not have dimension 2");
        return NULL;
    }
    //      - make sure stride is 4 in last dimension (ie, is C-style and contiguous)
    CHECKARRAYCARRAY(ary_coorda,"Array A");
    CHECKARRAYCARRAY(ary_coordb,"Array B");


    // Create return array containing Distances
    dim2[0] = arrayIndicesDims[0];
    dim2[1] = 1;
    ArrayDistances = (PyArrayObject*) PyArray_SimpleNew(1,dim2,NPY_FLOAT);
    Distances = (float*) PyArray_DATA(ArrayDistances);
    
    truestride=npaddedatoms*3;
    
    for (int i = 0; i < arrayIndicesDims[0]; i++) {
        float msd = ls_rmsd2_aligned_T_g(nrealatoms, npaddedatoms, rowstride, AData+Indices[i]*truestride, BData, GAData[Indices[i]], G_y);
        Distances[i] = sqrtf(msd);
    }
    
    return PyArray_Return(ArrayDistances);
}




static PyMethodDef _rmsd_methods[] = {
  {"getMultipleRMSDs_aligned_T_g", (PyCFunction)_getMultipleRMSDs_aligned_T_g, METH_VARARGS, "Theobald rmsd calculation on numpy-Tg."},
  {"getMultipleRMSDs_aligned_T_g_at_indices", (PyCFunction)_getMultipleRMSDs_aligned_T_g_at_indices, METH_VARARGS, "Theobald rmsd calculate on numpy Tg"},
  {"getMultipleRMSDs_aligned_T_g_at_indices_serial", (PyCFunction)_getMultipleRMSDs_aligned_T_g_at_indices_serial, METH_VARARGS, "Theobald rmsd calculate on numpy Tg"},
  
  {NULL, NULL, 0, NULL}
};

DL_EXPORT(void) init_rmsdcalc(void) {
  Py_InitModule3("_rmsdcalc", _rmsd_methods, "Numpy wrappers for fast Theobald rmsd calculation.");
  import_array();
}
