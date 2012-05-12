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

void drive_permute(int DIM, int STEP, int START, float *xa, float *xb, int *indices) {
  // Input: A bunch of indices.
  // Result: The indices are swapped.
  
  int dx, dy, dz;
  float x1, y1, z1;
  float x2, y2, z2;
  int dr2;
  // Conversion factor to go from nanometers to picometers
  int conv = 1000;
  int conv2 = conv*conv;
  // Threshold beyond which the pairwise distance is not computed at all
  int Thresh  = 1000;
  // A big number used to fill in the distance matrix elements that are thresholded out
  int BIG = conv2*3;
  // Allocate the distance matrix (integer).  The units are in squared picometers.
  int *DistanceMatrix = calloc(DIM*DIM,sizeof(int));
  int z_p;
  int INF = 2000000000;
  double start = get_time_precise();
  for (int i=0;i<DIM;i++) {
    x1 = xa[0*STEP+i+START];
    y1 = xa[1*STEP+i+START];
    z1 = xa[2*STEP+i+START];
    for (int j=0;j<DIM;j++) {
      x2 = xb[0*STEP+j+START];
      // The distance is converted to an integer number of picometers.
      dx = (int) conv*(x2 - x1);
      // At each step, if the x / y / z distance is bigger than the threshold (set to 1 nm)
      // then the distance matrix element is set to the BIG number.
      if (dx > Thresh)
	dr2 = BIG;
      else {
	y2 = xb[1*STEP+j+START];
	dy = (int) conv*(y2 - y1);
	if (dy > Thresh)
	  dr2 = BIG;
	else {
	  z2 = xb[2*STEP+j+START];
	  dz = (int) conv*(z2 - z1);
	  if (dz > Thresh)
	    dr2 = BIG;
	  else {
	    dr2 = (dx*dx + dy*dy + dz*dz);
	  }
	}
      }
      DistanceMatrix[i*DIM+j] = dr2;
    }
  }
  time_accumulate(&DistMatTime,start);

  //////////////////////////
  /// LPW Drive the APC! ///
  //////////////////////////
  start = get_time_precise();
  apc(DIM,DistanceMatrix,INF,&z_p,indices);
  time_accumulate(&APCTime,start);
  free(DistanceMatrix);
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
    3) Align using one set of atom labels and permutable atoms, and compute the RMSD using another set of labels + the same permutable atoms (combines 1 and 2)
    
    Conceptually there are three different sets of atomic indices:
    - AtomIndices are the distinguishable atoms used for alignment
    - PermuteIndices are identical and exchangeable atoms (more than one batch is possible)
    - AltIndices are the alternate labels used instead of the AtomIndices when computing the RMSD
    
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
    |  |  |- (Not implemented yet, but thinking): If the estimated final answer is larger than some lower bound, then don't go on.
    |  |  |- Rotate the atoms in (Atom+Permute)Indices using the rotation matrix
    +- If there are PermuteIndices:
    |  |- Permute the atomic indices for each batch in PermuteIndices (This is the bottleneck).
    |  |- Perform alignment using (Atom+Permute)Indices; this sets the RMSD values and gets the rotation matrices
    |  +- If output coordinates are requested, or if there are AltIndices:
    |  |  |- Rotate the whole frame using the rotation matrix
    |  |  |- Relabel the frame using the permutations
    |  |  +- If there are AltIndices:
    |  |  |  |- Loop through the AltIndices and the PermuteAtoms to get the RMSD explicitly
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
  int DebugPrint = 1;
  if (DebugPrint) {
    start = get_time_precise();
    printf("Preparing...\n");
  }

  /**********************************/
  /*   Initialize input variables   */
  /**********************************/
  // TheoData for distinguishable atoms
  PyArrayObject *XYZ_id_a_, *XYZ_id_b_, *G_id_a_;
  int nreal_id=-1,npad_id=-1,strd_id=-1;
  float G_id_b=-1;
  // TheoData for distinguishable+permutable atoms
  PyArrayObject *XYZ_lp_a_, *XYZ_lp_b_, *G_lp_a_;
  int nreal_lp=-1,npad_lp=-1,strd_lp=-1;
  float G_lp_b=-1;
  // Arrays for permutable indices and permutable atom 'batch' size (i.e. oxygens, hydrogens)
  PyArrayObject *LP_Flat_,*LP_Lens_;
  // Array for alternate indices
  PyArrayObject *Alt_Idx_;
  // Arrays for RMSD and rotation matrices
  PyArrayObject *RMSD_, *Rotations_;
  // The entire set of XYZ coordinates for fitting trajectory and reference frame
  PyArrayObject *XYZ_all_a_, *XYZ_all_b_;
  int Usage=-1;

  float msd;

  if (!PyArg_ParseTuple(args, "iiiiOOOfiiiOOOfOOOOOO", &Usage,
			&nreal_id, &npad_id, &strd_id, &XYZ_id_a_, &XYZ_id_b_, &G_id_a_, &G_id_b, 
			&nreal_lp, &npad_lp, &strd_lp, &XYZ_lp_a_, &XYZ_lp_b_, &G_lp_a_, &G_lp_b, 
			&LP_Flat_, &LP_Lens_, &Alt_Idx_, &Rotations_, &XYZ_all_a_, &XYZ_all_b_)) {
    printf("Mao says: Inputs / outputs not correctly specified!\n");
    return NULL;
  }

  /**********************************/
  /*   Initialize local variables   */
  /**********************************/
  // Settings for running this subroutine
  int HaveID = Usage / 1000;
  int HaveLP = (Usage % 1000) / 100;
  int HaveAlt = (Usage % 100) / 10;
  int WantXYZ = (Usage % 10) / 1;
  // The number of frames.
  int ns = XYZ_id_a_->dimensions[0];
  // Total number of atoms.
  int na_all = XYZ_all_a_->dimensions[2];
  // Number of distinguishable+permutable atoms (pad)
  int na_lp = XYZ_lp_a_->dimensions[2];
  // Number of permutable atom groups
  int n_lp_grps = LP_Lens_->dimensions[0];
  // Number of permutable (or alternate) atoms
  int totlen = LP_Flat_->dimensions[0];
  // Number of alternate index atoms
  int altlen = Alt_Idx_->dimensions[0];
  // Number of distinguishable atoms (true)
  int na_id = nreal_id;
  // TheoData for distinguishable atoms
  float *XYZ_id_a = (float*) XYZ_id_a_->data;
  float *XYZ_id_b = (float*) XYZ_id_b_->data;
  float *G_id_a = (float*) G_id_a_->data;
  // TheoData for distinguishable+permutable atoms
  float *XYZ_lp_a = (float*) XYZ_lp_a_->data;
  float *XYZ_lp_b = (float*) XYZ_lp_b_->data;
  float *G_lp_a = (float*) G_lp_a_->data;
  // Arrays for permutable indices and permutable atom group size
  long unsigned int *lp_lens = (long unsigned int*) LP_Lens_->data;
  long unsigned int *lp_flat = (long unsigned int*) LP_Flat_->data;
  // Arrays for alternate indices
  long unsigned int *alt_idx = (long unsigned int*) Alt_Idx_->data;
  // Arrays for RMSD and rotation matrices; allocate the RMSD array
  npy_intp dim2[2];
  dim2[0] = ns;
  dim2[1] = 1;
  RMSD_ = (PyArrayObject*) PyArray_SimpleNew(1,dim2,NPY_FLOAT);
  float *RMSD = (float*) PyArray_DATA(RMSD_);
  float *Rotations = (float*) Rotations_->data;
  // The entire set of XYZ coordinates for fitting trajectory and reference frame
  float *XYZ_all_a = (float*) XYZ_all_a_->data;
  float *XYZ_all_b = (float*) XYZ_all_b_->data;

  /********************************/
  /*     LPW Debug Printout       */
  /********************************/
  if (DebugPrint) {
    printf("HaveID = %i HaveLP = %i HaveAlt = %i WantXYZ = %i\n",HaveID,HaveLP,HaveAlt,WantXYZ);
    PrintDimensions("XYZ_id_a_", XYZ_id_a_);
    PrintDimensions("XYZ_id_b_", XYZ_id_b_);
    PrintDimensions("G_id_a_", G_id_a_);
    printf("nreal_id: %i, npad_id: %i, stride_id: %i\n",nreal_id,npad_id,strd_id);
    printf("\n");
    PrintDimensions("XYZ_lp_a_", XYZ_lp_a_);
    PrintDimensions("XYZ_lp_b_", XYZ_lp_b_);
    PrintDimensions("G_lp_a_", G_lp_a_);
    printf("nreal_lp: %i, npad_lp: %i, stride_lp: %i\n",nreal_lp,npad_lp,strd_lp);
    printf("\n");
    printf("Usage Mode: %i\n",Usage);
    PrintDimensions("Rotations_", Rotations_);
    PrintDimensions("XYZ_all_a_", XYZ_all_a_);
    PrintDimensions("XYZ_all_b_", XYZ_all_b_);
    PrintDimensions("LP_Lens_", LP_Lens_);
    PrintDimensions("LP_Flat_", LP_Flat_);
    printf("LP_Lens has this many dimensions: %i\n",LP_Lens_->nd);
  }


  /*********************************/
  /* Initialize internal variables */
  /*   (Memory is allocated here)  */
  /*********************************/
  // Temporary labels for old and new index
  int Old, New, StartIdx;
  int *lp_all, *lp_all_glob, *lp_starts;
  // The underscore d (f) means double (single) precision.
  // Double precision is needed for the dgemm routine (because sgemm appears to be broken!)
  double *X_lp_d, *Y_lp_d;
  float  *Y_lp_f, *Z_lp_f;
  double *X_all_d, *Y_all_d, *Z_all_d;
  double *X_alt_d, *Y_alt_d;
  // The 'truestride' for pad XYZ coordinates
  int true_id = npad_id*3;
  int true_lp = npad_lp*3;
  if (HaveLP) {
    // Create an array from the batch size which points to where the batches start
    // For example [5, 5, 6, 3] -> [0, 5, 10, 16], you know what i mean.
    lp_starts = calloc(n_lp_grps,sizeof(int));
    StartIdx = 0;
    for (int i = 0; i<n_lp_grps; i++) {
      lp_starts[i] = StartIdx;
      StartIdx += LP_Lens_->data[i];
      if (DebugPrint) {
	printf("Index Group %i : Starts at %i and has Length %i \n",i,lp_starts[i],*(lp_lens+i));
	for (int j = 0; j<*(lp_lens+i); j++)
	  printf("%i ",lp_flat[lp_starts[i] + j]);
	printf("\n");
      }
    }
    // These three are for rotating the distinguishable+permutable atoms using the pre-rotation matrix.
    X_lp_d = calloc(na_lp * ns * 3, sizeof(double));
    Y_lp_d = calloc(na_lp * ns * 3, sizeof(double));
    Y_lp_f = calloc(na_lp * ns * 3, sizeof(float));
    // This is for storing the distinguishable+permutable atoms with swapped coordinates.
    Z_lp_f = calloc(na_lp * ns * 3, sizeof(float));
    // An array for all of the permuted indices across all frames, counting from only the permutable indices or all indices.
    // For example, if the permutable atoms are 8, 10, and 12, these two arrays will be [0, 2, 1, 1, 0, 2] and [8, 12, 10, 10, 8, 12]
    lp_all = calloc(ns*totlen,sizeof(int));
    lp_all_glob = calloc(ns*totlen,sizeof(int));
    // Copy the permutable coordinates.
    for (int i=0; i<na_lp * ns * 3 ; i++) {
      X_lp_d[i] = (double) XYZ_lp_a[i]; 
      Z_lp_f[i] = (float) XYZ_lp_a[i];
    }
  }
  if (WantXYZ || (HaveLP && HaveAlt)) {
    // These three are for rotating the entire frame using the final rotation matrix.
    X_all_d = calloc(na_all * ns * 3, sizeof(double));
    Y_all_d = calloc(na_all * ns * 3, sizeof(double));
    // This is for storing the entire frame, rotated, with swapped coordinates.
    Z_all_d = calloc(na_all * ns * 3, sizeof(double));
    // Copy the single-precision coordinates into the double-precision coordinates
    for (int i=0; i<na_all * ns * 3 ; i++) {
      X_all_d[i] = (double) XYZ_all_a[i] ; 
    }
  }
  if (HaveAlt) {
    // The original and rotated alternate (ligand) coordinates.
    X_alt_d = calloc(altlen * ns * 3, sizeof(double));
    Y_alt_d = calloc(altlen * ns * 3, sizeof(double));
    // Copy the ligand coordinates.
    for (int i=0; i < ns; i++) {
      for (int j=0; j < altlen ; j++) {
	*(X_alt_d + i*altlen*3 + 0*altlen + j) = (double) *(XYZ_all_a + i*na_all*3 + 0*na_all + alt_idx[j]);
	*(X_alt_d + i*altlen*3 + 1*altlen + j) = (double) *(XYZ_all_a + i*na_all*3 + 1*na_all + alt_idx[j]);
	*(X_alt_d + i*altlen*3 + 2*altlen + j) = (double) *(XYZ_all_a + i*na_all*3 + 2*na_all + alt_idx[j]);
      }
    }
  }

  // Rotation matrix and dummy indices
  double rot[9];
  int j, k, p, Idx;
  if (DebugPrint) {
    time_accumulate(&dif,start);
    printf("Preparation stage done (% .4f seconds)\n",dif);
  }
  // Timing variables
  double RMSD1Time = 0.0;
  double MatrixTime = 0.0;
  double AltRMSDTime = 0.0;
  double RMSD2Time = 0.0;
  double PermuteTime = 0.0;
  double RelabelTime = 0.0;
  APCTime = 0.0;
  DistMatTime = 0.0;
  float x1, x2, y1, y2, z1, z2;

  /**********************************/
  /*   Start the RMSD calculation   */
  /*    Parallelized over frames    */
  /**********************************/
      
#pragma omp parallel for private(rot, msd, j, k, p, start, end, Idx, Old, New, x1, x2, y1, y2, z1, z2)
  for (int i = 0; i < ns; i++) 
    {
      //If there are AtomIndices:
      if (HaveID) {
	start = get_time_precise();
	// Perform alignment using AtomIndices; this sets the RMSD values and optionally gets the rotation matrices.
	ls_rmsd2_aligned_T_g(nreal_id,npad_id,strd_id,(XYZ_id_a+i*true_id),XYZ_id_b,G_id_a[i],G_id_b,&msd,(HaveLP || HaveAlt || WantXYZ),rot);
	time_accumulate(&RMSD1Time,start);
	// If there are no PermuteIndices:
	if (!HaveLP) {
	  start = get_time_precise();
	  // If we want output coordinates:
	  if (WantXYZ) {
	    // Rotate the whole frame using the rotation matrix
	    cblas_dgemm(101,112,111,3,na_all,3,1.0,rot,3,(X_all_d+i*3*na_all),na_all,0.0,(Y_all_d+i*3*na_all),na_all);
	    time_accumulate(&MatrixTime,start);
	  }
	  // If there are AltIndices:
	  if (HaveAlt) {
	    // Rotate the atoms in AltIndices using the rotation matrix.
	    cblas_dgemm(101,112,111,3,altlen,3,1.0,rot,3,(X_alt_d+i*3*altlen),altlen,0.0,(Y_alt_d+i*3*altlen),altlen);
	    time_accumulate(&MatrixTime,start);
	    start = get_time_precise();
	    msd = 0.0 ;
	    // Set the RMSD values by explicitly computing them from pairwise distances.
	    for (j=0; j<altlen; j++) {
	      x2 = Y_alt_d[(i*3+0)*altlen + j];
	      y2 = Y_alt_d[(i*3+1)*altlen + j];
	      z2 = Y_alt_d[(i*3+2)*altlen + j];
	      Idx = alt_idx[j];
	      x1 = XYZ_all_b[0*na_all + Idx];
	      y1 = XYZ_all_b[1*na_all + Idx];
	      z1 = XYZ_all_b[2*na_all + Idx];
	      msd = msd + (x2-x1)*(x2-x1);
	      msd = msd + (y2-y1)*(y2-y1);
	      msd = msd + (z2-z1)*(z2-z1);
	    }
	    msd = msd / altlen;
	    time_accumulate(&AltRMSDTime,start);
	  }
	}
	// If there are PermuteIndices:
	// Rotate the atoms in (Atom+Permute)Indices using the rotation matrix
	else {
	  cblas_dgemm(101,112,111,3,na_lp,3,1.0,rot,3,(X_lp_d+i*3*na_lp),na_lp,0.0,Y_lp_d+i*3*na_lp,na_lp);
	}
      }
      // If there are PermuteIndices:
      if (HaveLP) {
	start = get_time_precise();
	time_accumulate(&MatrixTime,start);
	for (k=0; k<na_lp * 3 ; k++) {
	  Y_lp_f[i*3*na_lp+k] = (float) Y_lp_d[i*3*na_lp+k];
	}
	// Permute the atomic indices for each batch in PermuteIndices.
	for (k=0; k<n_lp_grps ; k++) {
	  start = get_time_precise();
	  // This calls a subroutine that builds the cost matrix and solves the assignment problem!
	  drive_permute(lp_lens[k],na_lp,na_id+lp_starts[k],(Y_lp_f+i*3*na_lp),XYZ_lp_b,lp_all+i*totlen+lp_starts[k]);
	  time_accumulate(&PermuteTime,start);
	  start = get_time_precise();
	  // Relabel the atoms according to our brand new permutations
	  for (p=0; p<lp_lens[k] ; p++) {
	    Old = na_id + lp_starts[k] + p;
	    New = na_id + lp_starts[k] + *(lp_all+i*totlen+lp_starts[k]+p) ;
	    // Z_lp_f contains (atomindices + relabeled permuteindices) and is used for alignment.
	    Z_lp_f[(i*3+0)*na_lp + New] = XYZ_lp_a[(i*3+0)*na_lp + Old];
	    Z_lp_f[(i*3+1)*na_lp + New] = XYZ_lp_a[(i*3+1)*na_lp + Old];
	    Z_lp_f[(i*3+2)*na_lp + New] = XYZ_lp_a[(i*3+2)*na_lp + Old];
	    // Store the new labels.  This will be used for explicit RMSD computation (if both PermuteIndices and AltIndices are present)
	    if (WantXYZ || HaveAlt) {
	      Old = lp_flat[lp_starts[k] + p];
	      New = lp_flat[lp_starts[k] + *(lp_all+i*totlen+lp_starts[k]+p)];
	      lp_all_glob[i*totlen+lp_starts[k]+p] = New;
	    }
	  }
	  time_accumulate(&RelabelTime,start);
	}
	start = get_time_precise();
	// Perform alignment using (Atom+Permute)Indices; this sets the RMSD values and gets the rotation matrices
	ls_rmsd2_aligned_T_g(nreal_lp,npad_lp,strd_lp,(Z_lp_f+i*true_lp),XYZ_lp_b,G_lp_a[i],G_lp_b,&msd,1,rot);
	time_accumulate(&RMSD2Time,start);
	// If output coordinates are requested:
	if (WantXYZ || HaveAlt) {
	  // Rotate the whole frame using the rotation matrix
	  cblas_dgemm(101,112,111,3,na_all,3,1.0,rot,3,(X_all_d+i*3*na_all),na_all,0.0,(Y_all_d+i*3*na_all),na_all);
	  for (k=0; k<3*na_all ; k++) {
	    *(Z_all_d+i*3*na_all+k) = *(Y_all_d+i*3*na_all+k);
	  }
	  // Relabel the frame using the permutations
	  for (k=0; k<totlen; k++) {
	    Old = lp_flat[k];
	    New = lp_all_glob[i*totlen + k];
	    Z_all_d[(i*3+0)*na_all + New] = Y_all_d[(i*3+0)*na_all + Old];
	    Z_all_d[(i*3+1)*na_all + New] = Y_all_d[(i*3+1)*na_all + Old];
	    Z_all_d[(i*3+2)*na_all + New] = Y_all_d[(i*3+2)*na_all + Old];
	  }
	  if (HaveAlt) {
	    start = get_time_precise();
	    msd = 0.0 ;
	    // Set the RMSD values by explicitly computing them from pairwise distances.
	    for (j=0; j<altlen; j++) {
	      Idx = alt_idx[j];
	      // Y_all_d contains all of the atoms being fitted
	      x2 = Y_all_d[0*na_all + Idx];
	      y2 = Y_all_d[1*na_all + Idx];
	      z2 = Y_all_d[2*na_all + Idx];
	      // XYZ_all_b contains all of the reference atom positions
	      x1 = XYZ_all_b[0*na_all + Idx];
	      y1 = XYZ_all_b[1*na_all + Idx];
	      z1 = XYZ_all_b[2*na_all + Idx];
	      msd = msd + (x2-x1)*(x2-x1);
	      msd = msd + (y2-y1)*(y2-y1);
	      msd = msd + (z2-z1)*(z2-z1);
	    }
	    for (k=0; k<totlen; k++) {
	      Old = lp_flat[k];
	      New = lp_all_glob[i*totlen + k];
	      // Z_all_d contains all of the atoms being fitted, with relabeled atom indices
	      x2 = Z_all_d[(i*3+0)*na_all + New];
	      y2 = Z_all_d[(i*3+1)*na_all + New];
	      z2 = Z_all_d[(i*3+2)*na_all + New];
	      // XYZ_all_b contains all of the reference atom positions
	      x1 = XYZ_all_b[(i*3+0)*na_all + Old];
	      y1 = XYZ_all_b[(i*3+1)*na_all + Old];
	      z1 = XYZ_all_b[(i*3+2)*na_all + Old];
	      msd = msd + (x2-x1)*(x2-x1);
	      msd = msd + (y2-y1)*(y2-y1);
	      msd = msd + (z2-z1)*(z2-z1);
	    }
	    msd = msd / (altlen + totlen);
	    time_accumulate(&AltRMSDTime,start);
	  }
	}
      }
      // Assign the RMSD values to the RMSDOut array
      RMSD[i] = sqrtf(msd);
      if (WantXYZ || HaveAlt || HaveLP) {
	// Assign the rotation matrix to the RotOut array
	for (j=0; j<9; j++) {
	  *(Rotations + i*9 + j) = (float) rot[j];
	}
      }
    }

  // If output coordinates are requested:
  // Assign the rotated frames to the XYZOut array
  if (WantXYZ) {
    for (int i=0; i<na_all * ns * 3 ; i++) {
      if (HaveLP) 
	XYZ_all_a[i] = (double) Z_all_d[i] ; 
      else {
	XYZ_all_a[i] = (double) Y_all_d[i] ; 
      }
    }
  }

  if (DebugPrint) {
    printf("First RMSD: % .4f seconds\n",RMSD1Time);
    printf("Rotation: % .4f seconds\n",MatrixTime);
    printf("Ligand-RMSD: % .4f seconds\n",AltRMSDTime);
    printf("Permutation: % .4f seconds\n",PermuteTime);
    printf("Distance Part: % .4f seconds\n",DistMatTime);
    printf("APC Part: % .4f seconds\n",APCTime);
    printf("Relabeling: % .4f seconds\n",RelabelTime);
    printf("Second RMSD: % .4f seconds\n",RMSD2Time);
  }

  if (HaveLP) {
    free(lp_starts);
    free(X_lp_d);
    free(Y_lp_d);
    free(Y_lp_f);
    free(Z_lp_f);
    free(lp_all);
    free(lp_all_glob);
  }
  if (WantXYZ) {
    free(X_all_d);
    free(Y_all_d);
    free(Z_all_d);
  }
  if (HaveAlt) {
    free(X_alt_d);
    free(Y_alt_d);
  }

  return PyArray_Return(RMSD_);
}

static PyMethodDef _lprmsd_methods[] = {
  {"LPRMSD_Multipurpose", (PyCFunction)_LPRMSD_Multipurpose, METH_VARARGS, "Multipurpose permutation-invariant RMSD."},
  {NULL, NULL, 0, NULL}
};

DL_EXPORT(void) init_lprmsd(void)
{
  Py_InitModule3("_lprmsd", _lprmsd_methods, "Numpy wrappers for RMSD calculation with linear-programming atomic index permutation.");
  import_array();
}
