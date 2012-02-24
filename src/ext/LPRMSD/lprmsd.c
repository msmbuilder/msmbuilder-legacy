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
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "theobald_rmsd.h"
#include "apc.h"
#include <omp.h>
//#include <time.h
#include <sys/time.h>

#define CHECKARRAYFLOAT(ary,name) if (PyArray_TYPE(ary) != NPY_FLOAT32) {\
                                     PyErr_SetString(PyExc_ValueError,name" was not of type float32");\
                                     return NULL;\
                                 } 

#define CHECKARRAYINT(ary,name) if (PyArray_TYPE(ary) != NPY_INT64) {\
                                     PyErr_SetString(PyExc_ValueError,name" was not of type int");\
                                     return NULL;\
                                 } 

#define CHECKARRAYCARRAY(ary,name) if ((PyArray_FLAGS(ary) & NPY_CARRAY) != NPY_CARRAY) {\
                                       PyErr_SetString(PyExc_ValueError,name" was not a contiguous well-behaved array in C order");\
                                       return NULL;\
                                   } 

double APCTime = 0.0 ;
double DistMatTime = 0.0 ;

double get_time_precise() {
  struct timeval tim;
  double Answer;
  gettimeofday(&tim,NULL);
  Answer = tim.tv_sec + (tim.tv_usec/1000000.0);
  return Answer;
}

void time_accumulate(double *acc, double start) {
  double end = get_time_precise();
  *(acc) += (end-start);
}

static PyObject *_getPermutation(PyObject *self, PyObject *args) {
    npy_intp dim2[2];
    float *AData,*BData;
    int *IData;
    npy_intp *arrayADims,*arrayBDims,*arrayIDims,*arrayAStrides,*arrayBStrides,*arrayIStrides;
    PyArrayObject* ArrayDistances;
    float* Distances;
    int DIM=-1;
    PyArrayObject *ary_coorda, *ary_coordb, *ary_indices;
    int Stride=-1;

    /*
      LPW's interpretation of how this wurx.
      1) PyArg_ParseTuple loads in the data as ints, PyArrayObjects and floats.
      2) Float-arrays are assigned to the PyArrayObjects
      3) Dimensions for the arrays are Gleaned (still don't understand Strides)
      4) Sanity-checking on the PyArrayObjects
      5) Create PyArrayObject and corresponding float-array for the answer
      6) Call the Funk to obtain the answer (consider omp pragma)
      7) Return the answer.

      Okay, what do I need for permutations?
      1) We need to build our 'cost matrix', everything as integers, so need xyz coordinates
      2) We need to provide a vector with the optimal assignment and the associated cost
      3) Blorg, we can hijack the RMSD thing because it gives us the XYZ coordinates anyhow :)
     */
 
    // Copied directly from RMSD Stuff.
    if (!PyArg_ParseTuple(args, "iOOO",&DIM, &ary_coorda, &ary_coordb, &ary_indices)) {
      return NULL;
    }
    //////////////////////////////////
    /// LPW Below are formalities! ///
    //////////////////////////////////
    // Get pointers to array data
    AData  = (float*) PyArray_DATA(ary_coorda);
    BData  = (float*) PyArray_DATA(ary_coordb);
    IData  = (int*) PyArray_DATA(ary_indices);

    // Get dimensions of arrays (# molecules, maxlingos)
    // Note: strides are in BYTES, not INTs
    arrayADims = PyArray_DIMS(ary_coorda);
    arrayAStrides = PyArray_STRIDES(ary_coorda);
    arrayBDims = PyArray_DIMS(ary_coordb);
    arrayBStrides = PyArray_STRIDES(ary_coordb);
    arrayIDims = PyArray_DIMS(ary_indices);
    arrayIStrides = PyArray_STRIDES(ary_indices);

    // Do some sanity checking on array dimensions
    //      - make sure they are of float32 data type
    CHECKARRAYFLOAT(ary_coorda,"Array A");
    CHECKARRAYFLOAT(ary_coordb,"Array B");
    CHECKARRAYINT(ary_indices,"Index Array");

    //      - make sure lingo/count/mag arrays are 2d and are the same size in a set (ref/q)
    if (ary_coorda->nd != 2) {
        PyErr_SetString(PyExc_ValueError,"Array A did not have dimension 2");
	return NULL;
    }
    if (ary_coordb->nd != 2) {
        PyErr_SetString(PyExc_ValueError,"Array B did not have dimension 2");
        return NULL;
    }
    if (ary_indices->nd != 1) {
        PyErr_SetString(PyExc_ValueError,"Index Array did not have dimension 1");
        return NULL;
    }
    //      - make sure stride is 4 in last dimension (ie, is C-style and contiguous)
    //////////////////////////////////
    /// LPW Done with formalities! ///
    //////////////////////////////////

    Stride = arrayADims[0];
    dim2[0] = DIM;
    dim2[1] = DIM;
    PyArrayObject* ArrayDistanceMatrix;
    ArrayDistanceMatrix = (PyArrayObject*) PyArray_SimpleNew(2,dim2,NPY_INT);
    int *DistanceMatrix = PyArray_DATA(ArrayDistanceMatrix);
    float dx, dy, dz, dr2f;
    int dr2;
    int conv = 1000;
    int conv2 = conv*conv;
    int BIG = conv2*3;

    for (int i=0;i<DIM;i++) {
      int Idx = IData[2*i];
      for (int j=0;j<DIM;j++) {
	int Jdx = IData[2*j];
	// Okay, the distance matrix needs to be integer. :P
	// We'll first multiply the distance by 1000 so they're in picometers.
	dx = (AData[Idx*Stride+0]-BData[Jdx*Stride+0]);
	if (dx > 1000)
	  dr2 = BIG;
	else {
	  dy = (AData[Idx*Stride+1]-BData[Jdx*Stride+1]);
	  if (dy > 1000)
	    dr2 = BIG;
	  else {
	    dz = (AData[Idx*Stride+2]-BData[Jdx*Stride+2]);
	    if (dz > 1000)
	      dr2 = BIG;
	    else {
	      dr2f = conv2*(dx*dx + dy*dy + dz*dz);
	      dr2 = (int) dr2f;
	    }
	  }
	}
	DistanceMatrix[i*DIM+j] = dr2;
      }
    }

    //////////////////////////
    /// LPW Drive the APC! ///
    //////////////////////////
    dim2[0] = DIM;
    dim2[1] = 1;
    
    PyArrayObject* ArraySwaps = (PyArrayObject*) PyArray_SimpleNew(1,dim2,NPY_INT);
    int *swaps = PyArray_DATA(ArraySwaps);

    int z_p;
    int INF = 2000000000;
    int n=DIM;
    apc(n,DistanceMatrix,INF,&z_p,swaps);
    return PyArray_Return(ArraySwaps);
}

void drive_permute(int DIM, int STEP, int START, float *xa, float *xb, int *indices) {
  // Input: A bunch of indices.
  // Result: The indices are swapped.
  
  int dx, dy, dz;
  float x1, y1, z1;
  float x2, y2, z2;
  int dr2;
  int conv = 1000;
  int conv2 = conv*conv;
  int Thresh  = 1000;
  int BIG = conv2*3;
  int *DistanceMatrix = calloc(DIM*DIM,sizeof(int));
  int z_p;
  int INF = 2000000000;

  int Trash;
  /*
  printf("DIM %i STEP %i START %i\n", DIM, STEP, START);
  printf("Geometry 1\n");
  for (int i=0;i<DIM;i++) {
    printf("% .4f % .4f % .4f\n", xa[0*STEP+i+START], xa[1*STEP+i+START], xa[2*STEP+i+START]);
  }

  printf("Geometry 2\n");
  for (int i=0;i<DIM;i++) {
    printf("% .4f % .4f % .4f\n", xb[0*STEP+i+START], xb[1*STEP+i+START], xb[2*STEP+i+START]);
  }
  printf("\n");
  */
  double start = get_time_precise();
  for (int i=0;i<DIM;i++) {
    //int Idx = IData[2*i];
    x1 = xa[0*STEP+i+START];
    y1 = xa[1*STEP+i+START];
    z1 = xa[2*STEP+i+START];
    for (int j=0;j<DIM;j++) {
      x2 = xb[0*STEP+j+START];
      //int Jdx = IData[2*j];
      //dx = 
	
	//xa[0*STEP+i+START]
      dx = (int) conv*(x2 - x1);
      //dx = (AData[Idx*Stride+0]-BData[Jdx*Stride+0]);
      if (dx > Thresh)
	dr2 = BIG;
      else {
	//y2 = xb[1*STEP+i+START];
	//dy = y2 - y1;
	//dy = (AData[Idx*Stride+1]-BData[Jdx*Stride+1]);
	y2 = xb[1*STEP+j+START];
	dy = (int) conv*(y2 - y1);
	if (dy > Thresh)
	  dr2 = BIG;
	else {
	  //z2 = xb[2*STEP+i+START];
	  //dz = z2 - z1;
	  //dz = (AData[Idx*Stride+2]-BData[Jdx*Stride+2]);
	  z2 = xb[2*STEP+j+START];
	  dz = (int) conv*(z2 - z1);
	  if (dz > Thresh)
	    dr2 = BIG;
	  else {
	    dr2 = (dx*dx + dy*dy + dz*dz);
	    //dr2 = (int) dr2f;
	  }
	}
      }
      DistanceMatrix[i*DIM+j] = dr2;
      //printf("%6i ", DistanceMatrix[i*DIM+j] );
    }
    //printf("\n");
  }
  time_accumulate(&DistMatTime,start);

  //Trash = system("read"); 

  //////////////////////////
  /// LPW Drive the APC! ///
  //////////////////////////

  // The end result should be 
  start = get_time_precise();
  apc(DIM,DistanceMatrix,INF,&z_p,indices);
  time_accumulate(&APCTime,start);
  free(DistanceMatrix);
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
    CHECKARRAYFLOAT(ary_coorda,"Array A");
    CHECKARRAYFLOAT(ary_coordb,"Array B");

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

    // LPW The size of dim2 gets increased by a factor of 10 if we want the rotation matrices. :)
    // Create return array containing Distances
    int NeedMat = 1;
    if (NeedMat)
      dim2[0] = arrayADims[0]*10;
    else
      dim2[0] = arrayADims[0];
    dim2[1] = 1;
    ArrayDistances = (PyArrayObject*) PyArray_SimpleNew(1,dim2,NPY_FLOAT);
    Distances = (float*) PyArray_DATA(ArrayDistances);

    truestride=npaddedatoms*3;

    float msd = 0.0;
    float *DistPtr;
    double rot[9];

#pragma omp parallel for private(rot)
    for (int i = 0; i < arrayADims[0]; i++) 
      {
	//DistPtr = Distances + arrayADims[0] + i*9;
	ls_rmsd2_aligned_T_g(nrealatoms,npaddedatoms,rowstride,(AData+i*truestride),BData,GAData[i],G_y,&msd,NeedMat,rot);
        Distances[i] = sqrtf(msd);
	//if (NeedMat)
	for (int j=0; j<9; j++) {
	  Distances[arrayADims[0] + i*9 + j] = rot[j];
	}
      }
    
    return PyArray_Return(ArrayDistances);
}

void PrintDimensions(char *title, PyArrayObject *array) {
  printf ("Dimensions of %s : (", title);
  int j, k;
  for (int i=0; i < array->nd; i++) {
    j = (int) array->dimensions[i];
    k = (int) array->strides[i];
    printf ("%i [stride %i]", j, k);
    if (i < (array->nd - 1)) {
      printf(", ");
    }
  }
  printf (")\n");
}

static PyObject *_LPRMSD_Multipurpose(PyObject *self, PyObject *args) {
  /*
    The all-purpose routine for permutation-invariant and alternate-indexed RMSD.
    Written by Lee-Ping Wang, 2012.
    
    This routine was written with these two capabilities in mind:
    1) Permute sets of identical and exchangable atom labels to minimize the RMSD (lets us build MSMs with explicit solvent!)
    2) Align using one set of atom labels (i.e. protein) and compute the RMSD using another set of labels (i.e. ligand), motivated by Morgan's projects
    
    Conceptually there are three different sets of atomic indices:
    - AtomIndices are the distinguishable atoms used for alignment
    - PermuteIndices are identical and exchangeable atoms (more than one batch is possible)
    - AltIndices are the alternate labels used for RMSD when using capability #2
    
    Note that I can't use PermuteIndices and AltIndices at the same time because that would just be ridiculous.

    There are several ways in which a user might run this:
    +- If there are AtomIndices:
    |  |- Perform alignment using AtomIndices; this sets the RMSD values and optionally gets the rotation matrices.
    |  +- If there are no PermuteIndices:
    |  |  +- If there are AltIndices:
    |  |  |  |- Rotate the atoms in AltIndices using the rotation matrix.
    |  |  |  |- Set the RMSD values by explicitly computing them from pairwise distances.
    |  |  +- If we want output coordinates:
    |  |  |  |- Rotate the whole frame using the rotation matrix
    |  |  |  |- Assign the rotation matrix to the RotOut array
    |  +- If there are PermuteIndices:
    |  |  |- (Considering this): If the estimated final answer is larger than some lower bound, then don't go on.
    |  |  |- Rotate the atoms in (Atom+Permute)Indices using the rotation matrix
    +- If there are PermuteIndices:
    |  |- Permute the atomic indices for each batch in PermuteIndices (This is the bottleneck).
    |  |- Perform alignment using (Atom+Permute)Indices; this sets the RMSD values and gets the rotation matrices
    |  +- If output coordinates are requested:
    |  |  |- Rotate the whole frame using the rotation matrix
    |  |  |- (If desired) Relabel the frame using the permutations
    +- If output coordinates are requested:
    |  |- Assign the rotated frame to the XYZOut array
    |- Assign the RMSD values to the RMSDOut array
    Done!!!

    The options (codified in the Usage integer):
    - Whether we have a set of AtomicIndices
    - Whether we have a set of AltIndices
    - Whether we have a set of PermuteIndices
    - Whether we want the output coordinates
     
    The arguments:
    - Input: Flag for usage mode (1 integer)
    - Input: TheoData for the AtomIndices (7 variables)
    - Input: TheoData for the AtomIndices+PermuteAtoms (7 variables)
    - Input: The set of PermuteIndices / AltIndices (1 array)
    - Input: An array of batch sizes in PermuteIndices (1 array)
    - Output (pointer): Rotation matrices (1 variable, size NShots * 9)
    - Input/Output (pointer): Entire trajectory (1 array, size NShots * 3 * NAtoms)
    - Input: The trajectory frame for the reference coordinate (1 array, size 3 * NAtoms)
    - Output (return): RMSD array  (1 variable, size NShots)

    

  */
  struct timeval tv;
  double start, end, dif;

  start = get_time_precise();

  printf("Preparing...\n");

  PyArrayObject *XYZData_id_a_, *XYZData_id_b_, *G_id_a_;
  int nrealatoms_id=-1,npaddedatoms_id=-1,rowstride_id=-1;
  float G_id_y=-1;

  PyArrayObject *XYZData_lp_a_, *XYZData_lp_b_, *G_lp_a_;
  int nrealatoms_lp=-1,npaddedatoms_lp=-1,rowstride_lp=-1;
  float G_lp_y=-1;

  PyArrayObject *LP_Flat_,*LP_Lens_, *Out_RMSD_, *Out_Rotations_, *Out_XYZAll_, *Ref_XYZAll_;
  int Usage=-1;
  
  float *RMSDOut, *RotOut, *AData_id, *BData_id, *GAData_id, *XYZOut, *XYZRef;
  float *AData_lp, *BData_lp, *GAData_lp;
  float msd;
  float *R_lp_f;
  PyArrayObject* ArrayDistances;
  npy_intp dim2[2];

  int Trash;

  if (!PyArg_ParseTuple(args, "iiiiOOOfiiiOOOfOOOOO", &Usage,
			&nrealatoms_id, &npaddedatoms_id, &rowstride_id,
			&XYZData_id_a_, &XYZData_id_b_, &G_id_a_, &G_id_y, 
			&nrealatoms_lp, &npaddedatoms_lp, &rowstride_lp,
			&XYZData_lp_a_, &XYZData_lp_b_, &G_lp_a_, &G_lp_y, 
			&LP_Flat_, &LP_Lens_, &Out_Rotations_, &Out_XYZAll_, &Ref_XYZAll_)) {
    printf("Mao says: Inputs / outputs not correctly specified!\n");
    return NULL;
  }

  /********************************/
  /*   LPW Obtain the settings    */
  /********************************/
  int HaveID = Usage / 1000;
  int HaveLP = (Usage % 1000) / 100;
  int Altidx = (Usage % 100) / 10;
  int WantXYZ = (Usage % 10) / 1;
  printf("HaveID = %i HaveLP = %i Altidx = %i WantXYZ = %i\n",HaveID,HaveLP,Altidx,WantXYZ);

  RotOut = (float*) Out_Rotations_->data;
  AData_id = (float*) XYZData_id_a_->data;
  BData_id = (float*) XYZData_id_b_->data;
  GAData_id = (float*) G_id_a_->data;
  AData_lp = (float*) XYZData_lp_a_->data;
  BData_lp = (float*) XYZData_lp_b_->data;
  GAData_lp = (float*) G_lp_a_->data;
  XYZOut = (float*) Out_XYZAll_->data;
  XYZRef = (float*) Ref_XYZAll_->data;

  R_lp_f = (float*) XYZData_lp_b_->data;
  //R_lp_f = (float*) XYZData_lp_b_->data;

  /********************************/
  /*     LPW Debug Printout       */
  /********************************/
  PrintDimensions("XYZData_id_a_", XYZData_id_a_);
  PrintDimensions("XYZData_id_b_", XYZData_id_b_);
  PrintDimensions("G_id_a_", G_id_a_);
  printf("nreal_id: %i, npadded_id: %i, stride_id: %i\n",nrealatoms_id,npaddedatoms_id,rowstride_id);

  printf("\n");

  PrintDimensions("XYZData_lp_a_", XYZData_lp_a_);
  PrintDimensions("XYZData_lp_b_", XYZData_lp_b_);
  PrintDimensions("G_lp_a_", G_lp_a_);
  printf("nreal_lp: %i, npadded_lp: %i, stride_lp: %i\n",nrealatoms_lp,npaddedatoms_lp,rowstride_lp);

  printf("\n");
  printf("Usage Mode: %i\n",Usage);
  PrintDimensions("Out_Rotations_", Out_Rotations_);
  PrintDimensions("Out_XYZAll_", Out_XYZAll_);
  PrintDimensions("Ref_XYZAll_", Ref_XYZAll_);
  PrintDimensions("LP_Lens_", LP_Lens_);
  PrintDimensions("LP_Flat_", LP_Flat_);
  
  /********************************/
  /*   LPW End Debug Printout     */
  /********************************/

  // The number of frames.
  int ns = XYZData_id_a_->dimensions[0];
  int na_all = Out_XYZAll_->dimensions[2];
  int na_lp = XYZData_lp_a_->dimensions[2];
  int na_id = nrealatoms_id;
  printf("LP_Lens has this many dimensions: %i\n",LP_Lens_->nd);

  int n_lp_grps = LP_Lens_->dimensions[0];
  long unsigned int *lp_lens = (long unsigned int*) LP_Lens_->data;
  long unsigned int *lp_flat = (long unsigned int*) LP_Flat_->data;
  int *lp_starts = calloc(n_lp_grps,sizeof(int));
  int StartIdx = 0;
  for (int i = 0; i<n_lp_grps; i++) {
    lp_starts[i] = StartIdx;
    StartIdx += LP_Lens_->data[i];
    printf("Index Group %i : Starts at %i and has Length %i \n",i,lp_starts[i],*(lp_lens+i));
    for (int j = 0; j<*(lp_lens+i); j++) {
      printf("%i ",lp_flat[lp_starts[i] + j]);
    }
    printf("\n");
  }

  // Internal variable naming

  float *X_lp_f = (float*) XYZData_lp_a_->data;

  double *X_all_d = calloc(na_all * ns * 3, sizeof(double));
  double *X_lp_d = calloc(na_lp * ns * 3, sizeof(double));

  double *Y_all_d = calloc(na_all * ns * 3, sizeof(double));
  double *Z_all_d = calloc(na_all * ns * 3, sizeof(double));
  double *Y_lp_d = calloc(na_lp * ns * 3, sizeof(double));
  float *Y_lp_f = calloc(na_lp * ns * 3, sizeof(float));
  float *Z_lp_f = calloc(na_lp * ns * 3, sizeof(float));


  int Old, New;
  
  int *lp_lens_1 = calloc(n_lp_grps,sizeof(int));
  int totlen = 0;
  for (int i = 0; i<n_lp_grps; i++) {
    lp_lens_1[i] = (int) lp_lens[i];
    totlen += lp_lens_1[i];
  }
  int *lp_all = calloc(ns*totlen,sizeof(int));
  int *lp_all_glob = calloc(ns*totlen,sizeof(int));

  double *L_all_d = calloc(totlen * ns * 3, sizeof(double));
  double *M_all_d = calloc(totlen * ns * 3, sizeof(double));
  //double *X1_lp = calloc(na_lp * ns * 3, sizeof(double));

  for (int i=0; i<na_all * ns * 3 ; i++) {
    X_all_d[i] = (double) XYZOut[i] ; 
  }

  for (int i=0; i < ns; i++) {
    for (int j=0; j < totlen ; j++) {
      *(L_all_d + i*totlen*3 + 0*totlen + j) = *(X_all_d + i*na_all*3 + 0*na_all + lp_flat[j]);
      *(L_all_d + i*totlen*3 + 1*totlen + j) = *(X_all_d + i*na_all*3 + 1*na_all + lp_flat[j]);
      *(L_all_d + i*totlen*3 + 2*totlen + j) = *(X_all_d + i*na_all*3 + 2*na_all + lp_flat[j]);
    }
  }
  

  for (int i=0; i<na_lp * ns * 3 ; i++) {
    X_lp_d[i] = (double) X_lp_f[i]; 
    Z_lp_f[i] = (float) AData_lp[i];
  }

  int truestride_id = npaddedatoms_id*3;
  int truestride_lp = npaddedatoms_lp*3;
  
  dim2[0] = ns;
  dim2[1] = 1;
  ArrayDistances = (PyArrayObject*) PyArray_SimpleNew(1,dim2,NPY_FLOAT);
  RMSDOut = (float*) PyArray_DATA(ArrayDistances);

  double rot[9];
  int j, k, p;
  //double *y_lp_d = calloc(na_lp * 3, sizeof(double));
  //double *y_lp_f = calloc(na_lp * 3, sizeof(float));
  gettimeofday(&tv,NULL);
  end = get_time_precise();
  //end = clock();
  //clock (&end);
  dif = end - start;
  
  printf("Preparation stage done (% .4f seconds)\n",dif);

  double RMSD1Time = 0.0;
  double MatrixTime = 0.0;
  double AltRMSDTime = 0.0;
  double RMSD2Time = 0.0;
  double PermuteTime = 0.0;
  double RelabelTime = 0.0;
  APCTime = 0.0;
  DistMatTime = 0.0;
  double msd1 = 0.0;

  int Idx;
  float x1, x2, y1, y2, z1, z2;

#pragma omp parallel for private(rot, msd, j, k, p, start, end, Idx, x1, x2, y1, y2, z1, z2)
  for (int i = 0; i < ns; i++) 
    {
  /*
    The all-purpose driver for permutation-invariant RMSD.
    Function: Align a set of coordinates to a single coordinate
    There are several ways in which a user might run this:
    +- If there exists a set of AtomIndices:
    |  |- Perform alignment using AtomIndices
    |  +- If there are no PermuteIndices:
    |  |  |- Assign the RMSD values to the RMSDOut array
    |  |  +- If output coordinates are requested:
    |  |  |  |- Rotate the whole frame using the rotation matrix
    |  |  |  |- Assign the rotation matrix to the RotOut array
    |  |  |  |- Assign the rotated frame to the XYZOut array
    |  +- If there are PermuteIndices:
    |  |  |- Rotate the A+P Coordinates using the rotation matrix
    +- If there are PermuteIndices:
    |  |- Permute the atomic indices.
    |  |- Perform alignment using A+P Coordinates
    |  |- Assign the RMSD values to the RMSDOut array
    |  +- If output coordinates are requested:
    |  |  |- Rotate the whole frame using the rotation matrix
    |  |  |- (If desired) Relabel the frame using the permutations
    |  |  |- Assign the rotated frame to the XYZOut array
    Done!!!
  */
      if (HaveID) {
	start = get_time_precise();
	ls_rmsd2_aligned_T_g(nrealatoms_id,npaddedatoms_id,rowstride_id,
			     (AData_id+i*truestride_id),BData_id,GAData_id[i],G_id_y,&msd,(HaveLP || Altidx || WantXYZ),rot);
	msd1 = msd;
	time_accumulate(&RMSD1Time,start);
	if (!HaveLP) {
	  if (WantXYZ || Altidx) {
	    for (j=0; j<9; j++) {
	      *(RotOut + i*9 + j) = (float) rot[j];
	    }
	    start = get_time_precise();
	    if (WantXYZ)
	      cblas_dgemm(101,112,111,3,na_all,3,1.0,rot,3,(X_all_d+i*3*na_all),na_all,0.0,(Y_all_d+i*3*na_all),na_all);
	    if (Altidx)
	      cblas_dgemm(101,112,111,3,totlen,3,1.0,rot,3,(L_all_d+i*3*totlen),totlen,0.0,(M_all_d+i*3*totlen),totlen);
	    time_accumulate(&MatrixTime,start);
	    start = get_time_precise();
	    msd = 0.0 ;
	    for (j=0; j<lp_lens_1[0]; j++) {
	      x2 = M_all_d[(i*3+0)*totlen + j];
	      y2 = M_all_d[(i*3+1)*totlen + j];
	      z2 = M_all_d[(i*3+2)*totlen + j];
	      Idx = lp_flat[j];
	      x1 = XYZRef[0*na_all + Idx];
	      y1 = XYZRef[1*na_all + Idx];
	      z1 = XYZRef[2*na_all + Idx];
	      msd = msd + (x2-x1)*(x2-x1);
	      msd = msd + (y2-y1)*(y2-y1);
	      msd = msd + (z2-z1)*(z2-z1);
	      //printf("x1 y1 z1 = % .4f % .4f % .4f\n",x1,y1,z1);
	      //printf("x2 y2 z2 = % .4f % .4f % .4f\n",x2,y2,z2);
	    }
	    //system("read");
	    msd = msd / lp_lens_1[0];
	    time_accumulate(&AltRMSDTime,start);
	  }
	}
      }
      if (HaveLP) {
	for (j=0; j<9; j++) {
	  *(RotOut + i*9 + j) = (float) rot[j];
	}
	start = get_time_precise();
	cblas_dgemm(101,112,111,3,na_lp,3,1.0,rot,3,(X_lp_d+i*3*na_lp),na_lp,0.0,Y_lp_d+i*3*na_lp,na_lp);
	time_accumulate(&MatrixTime,start);
	for (k=0; k<na_lp * 3 ; k++) {
	  Y_lp_f[i*3*na_lp+k] = (float) Y_lp_d[i*3*na_lp+k];
	}

	for (k=0; k<n_lp_grps ; k++) {
	  start = get_time_precise();
	  drive_permute(lp_lens_1[k],na_lp,na_id + lp_starts[k],Y_lp_f+i*3*na_lp,R_lp_f,lp_all+i*totlen+lp_starts[k]);
	  time_accumulate(&PermuteTime,start);
	  start = get_time_precise();
	  for (p=0; p<lp_lens_1[k] ; p++) {
	    Old = na_id + lp_starts[k] + p;
	    New = na_id + lp_starts[k] + *(lp_all+i*totlen+lp_starts[k]+p) ;
	    Z_lp_f[(i*3+0)*na_lp + New] = AData_lp[(i*3+0)*na_lp + Old];
	    Z_lp_f[(i*3+1)*na_lp + New] = AData_lp[(i*3+1)*na_lp + Old];
	    Z_lp_f[(i*3+2)*na_lp + New] = AData_lp[(i*3+2)*na_lp + Old];

	    if (WantXYZ) {
	      Old = lp_flat[lp_starts[k] + p];
	      New = lp_flat[lp_starts[k] + *(lp_all+i*totlen+lp_starts[k]+p)];
	      lp_all_glob[i*totlen+lp_starts[k]+p] = New;
	    }
	  }
	  time_accumulate(&RelabelTime,start);
	}

	start = get_time_precise();
	ls_rmsd2_aligned_T_g(nrealatoms_lp,npaddedatoms_lp,rowstride_lp,
			     (Z_lp_f+i*truestride_lp),BData_lp,GAData_lp[i],G_lp_y,&msd,WantXYZ,rot);
	time_accumulate(&RMSD2Time,start);

	if (WantXYZ) {
	  cblas_dgemm(101,112,111,3,na_all,3,1.0,rot,3,(X_all_d+i*3*na_all),na_all,0.0,(Y_all_d+i*3*na_all),na_all);
	  for (k=0; k<3*na_all ; k++) {
	    *(Z_all_d+i*3*na_all+k) = *(Y_all_d+i*3*na_all+k);
	  }
	  for (k=0; k<totlen; k++) {
	    Old = lp_flat[k];
	    New = lp_all_glob[i*totlen + k];
	    Z_all_d[(i*3+0)*na_all + New] = Y_all_d[(i*3+0)*na_all + Old];
	    Z_all_d[(i*3+1)*na_all + New] = Y_all_d[(i*3+1)*na_all + Old];
	    Z_all_d[(i*3+2)*na_all + New] = Y_all_d[(i*3+2)*na_all + Old];
	  }
	}
      }
      RMSDOut[i] = sqrtf(msd);
    }

  if (WantXYZ) {
    for (int i=0; i<na_all * ns * 3 ; i++) {
      if (HaveLP) 
	XYZOut[i] = (double) Z_all_d[i] ; 
      else {
	XYZOut[i] = (double) Y_all_d[i] ; 
      }
    }
  }

  printf("First RMSD: % .4f seconds\n",RMSD1Time);
  printf("Rotation: % .4f seconds\n",MatrixTime);
  printf("Ligand-RMSD: % .4f seconds\n",AltRMSDTime);
  printf("Permutation: % .4f seconds\n",PermuteTime);
  printf("Distance Part: % .4f seconds\n",DistMatTime);
  printf("APC Part: % .4f seconds\n",APCTime);
  printf("Relabeling: % .4f seconds\n",RelabelTime);
  printf("Second RMSD: % .4f seconds\n",RMSD2Time);


  free(lp_starts);
  free(X_all_d);
  free(X_lp_d);
  free(Y_all_d);
  free(Z_all_d);
  free(Y_lp_d);
  free(Y_lp_f);
  free(Z_lp_f);
  free(lp_lens_1);
  free(lp_all);
  free(lp_all_glob);

  return PyArray_Return(ArrayDistances);
}

static PyMethodDef _lprmsd_methods[] = {
  {"getPermutation", (PyCFunction)_getPermutation, METH_VARARGS, "Atomic index permutation method."},
  {"getMultipleRMSDs_aligned_T_g", (PyCFunction)_getMultipleRMSDs_aligned_T_g, METH_VARARGS, "Theobald rmsd calculation on numpy-Tg."},
  {"LPRMSD_Multipurpose", (PyCFunction)_LPRMSD_Multipurpose, METH_VARARGS, "Multipurpose permutation-invariant RMSD."},
  {NULL, NULL, 0, NULL}
};

DL_EXPORT(void) init_lprmsd(void)
{
  Py_InitModule3("_lprmsd", _lprmsd_methods, "Numpy wrappers for RMSD calculation with linear-programming atomic index permutation.");
  import_array();
}
