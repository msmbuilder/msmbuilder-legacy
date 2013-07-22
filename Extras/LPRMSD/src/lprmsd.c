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

#define CHECKARRAYFLOAT(ary,name) if (PyArray_TYPE(ary) != NPY_FLOAT32) { \
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

int compare (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

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

void drive_permute(int DIM, int STEP, int START, float *xa, float *xb, int *indices, int Subset) {
  // Input: A bunch of indices.
  // Result: The indices are swapped.
  
  int dx, dy, dz;
  float x1, y1, z1;
  float x2, y2, z2;
  int dr2;
  int bRect = (Subset != 0 && Subset < DIM);
  // Conversion factor to go from nanometers to picometers
  int conv = 1000;
  int conv2 = conv*conv;
  // Threshold beyond which the pairwise distance is not computed at all
  int Thresh  = 1000;
  // A big number used to fill in the distance matrix elements that are thresholded out
  int BIG = conv2*3;
  // Allocate the distance matrix (integer).  The units are in squared picometers.
  int *DistanceMatrix, *DistanceMatrixA, *DistanceMatrixB;
  int *indicesA, *indicesBT, *indicesB, *indicesCount;
  int *FillIdx;
  int *FillDist;
  int *FillSort;
  DistanceMatrix = calloc(DIM*DIM,sizeof(int));
  if (bRect) {
    DistanceMatrixA = calloc(DIM*DIM,sizeof(int));
    DistanceMatrixB = calloc(DIM*DIM,sizeof(int));
    indicesA  = calloc(DIM,sizeof(int));
    indicesB  = calloc(DIM,sizeof(int));
    indicesBT = calloc(DIM,sizeof(int));
    indicesCount = calloc(DIM,sizeof(int));
    FillSort = calloc(DIM,sizeof(int));
    FillDist = calloc(DIM,sizeof(int));
    FillIdx  = calloc(DIM,sizeof(int));
  }
  int ENum = 0;

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
      if (bRect) {
	if (i < Subset) {
	  DistanceMatrixA[j*DIM+i] = dr2;
	}
	if (j < Subset) {
	  DistanceMatrixB[i*DIM+j] = dr2;
	}
      }
      DistanceMatrix[j*DIM+i] = dr2;
    }
  }
  time_accumulate(&DistMatTime,start);

  //////////////////////////
  /// LPW Drive the APC! ///
  //////////////////////////
  start = get_time_precise();
  double ElementA, ElementB;
  int MapCount = 0;
  int Fill = 0;
  if (bRect) {
    apc(DIM,DistanceMatrixA,INF,&z_p,indicesA);
    apc(DIM,DistanceMatrixB,INF,&z_p,indicesBT);
    for (int i=0; i<DIM; i++)
      indicesB[indicesBT[i]] = i;
    for (int i=0; i<DIM; i++) {
      ElementA = DistanceMatrixA[i*DIM + indicesA[i]];
      ElementB = DistanceMatrixB[indicesB[i]*DIM + i];
      //printf("indicesA[%3i] = %3i ElementA = %.4f indicesB[%3i] = %3i ElementB = %.4f\n", i, indicesA[i], sqrtf((double)ElementA/conv2), i, indicesB[i], sqrtf((double)ElementB/conv2));
      if (i < Subset && indicesA[i] == indicesB[i]) {
	indices[i] = indicesA[i] ; 
	//printf("Setting indices[%i] to %i", i, indices[i]);
	//printf("Incrementing indicesCount[%i]\n",indices[i]);
	indicesCount[indices[i]]++;
	MapCount++;
      } else {
	indices[i] = -1;
	if (Fill) {
	  if (ElementA != 0) {
	    FillDist[ENum] = ElementA;
	    FillIdx[ENum] = i*DIM + indicesA[i];
	    //printf("FillDist[%i] = %i FillIdx[%i] = %i,%i\n",ENum,FillDist[ENum],ENum,FillIdx[ENum]/DIM,FillIdx[ENum]%DIM);
	    ENum++;
	  } 
	  if (ElementB != 0) {
	    FillDist[ENum] = ElementB;
	    FillIdx[ENum] = i*DIM + indicesB[i];
	    //printf("FillDist[%i] = %i FillIdx[%i] = %i,%i\n",ENum,FillDist[ENum],ENum,FillIdx[ENum]/DIM,FillIdx[ENum]%DIM);
	    ENum++;
	  }
	}
      }
    }

    if (Fill) {
      // This is a fill based on the atom pairs that have the lowest RMSD.
      /*
      for (int i=0; i<DIM; i++) {
	//printf("Setting FillSort[%i] to %i\n", i, FillDist[i]);
	FillSort[i] = FillDist[i];
      }
      qsort(FillSort, Subset, sizeof(int), compare);
      for (int i=0; i<DIM; i++) {
	if (FillSort[i] != 0) {
	  for (int j=0; j<DIM; j++) {
	    if (MapCount == Subset)
	      break;
	    if (FillSort[i] == FillDist[j]) {
	      printf("DistanceMatrix[%i,%i] has distance % .4f\n",FillIdx[j]/DIM,FillIdx[j]%DIM,sqrtf((double)DistanceMatrix[FillIdx[j]] / conv2));
	      if (indices[FillIdx[j]/DIM] == -1) {
		indices[FillIdx[j]/DIM] = FillIdx[j]%DIM;
		printf("Filling indices[%i] = %i\n",FillIdx[j]/DIM, FillIdx[j]%DIM);
		MapCount++;
	      } 
	    }
	  }
	}
      }
      */
      // This is a fill based on the atom ordering.
      for (int i=0; i<DIM; i++) {
	if (MapCount == Subset)
	  break;
	if (FillDist[i] != 0 && indicesCount[FillIdx[i]%DIM] == 0) {
	  //printf("indicesCount[%i] = %i\n", FillIdx[i]%DIM, indicesCount[FillIdx[i]%DIM]);
	  //printf("DistanceMatrix[%i,%i] has distance % .4f\n",FillIdx[i]/DIM,FillIdx[i]%DIM,sqrtf((double)DistanceMatrix[FillIdx[i]] / conv2));
	  if (indices[FillIdx[i]/DIM] == -1) {
	    indices[FillIdx[i]/DIM] = FillIdx[i]%DIM;
	    indicesCount[FillIdx[i]%DIM]++;
	    //printf("Filling indices[%i] = %i and incrementing indicesCount[%i]\n",FillIdx[i]/DIM, FillIdx[i]%DIM,FillIdx[i]%DIM);
	    MapCount++;
	  } else if (FillDist[i] < DistanceMatrix[DIM * (FillIdx[i]/DIM) + indices[FillIdx[i]/DIM]]) {
	    //printf("Replacing indices[%i] = %i with %i\n", FillIdx[i]/DIM, indices[FillIdx[i]/DIM], FillIdx[i]%DIM);
	    //printf("DistanceMatrix[%i, %i (%i) ] = % .4f\n", FillIdx[i]/DIM, indices[FillIdx[i]/DIM], DIM * (FillIdx[i]/DIM) + indices[FillIdx[i]/DIM], sqrtf((double)DistanceMatrix[DIM * (FillIdx[i]/DIM) + indices[FillIdx[i]/DIM]] / conv2));
	    //printf("DistanceMatrix[%i, %i (%i) ] = % .4f or % .4f\n", FillIdx[i]/DIM, FillIdx[i]%DIM, FillIdx[i], sqrtf((double)DistanceMatrix[FillIdx[i]] / conv2), sqrtf((double)FillDist[i] / conv2));
	    //printf("Incrementing indicesCount[%i] and decrementing indicesCount[%i]\n",FillIdx[i]%DIM,indices[FillIdx[i]/DIM]);
	    indicesCount[indices[FillIdx[i]/DIM]]--;
	    indices[FillIdx[i]/DIM] = FillIdx[i]%DIM;
	    indicesCount[FillIdx[i]%DIM]++;
	  } else {
	    //printf("Not filling indices[%i] with %i because current value is equal to %i\n", FillIdx[i]/DIM, FillIdx[i]%DIM, indices[FillIdx[i]/DIM]);
	  }
	}
      }
    }

    int Index1 = 0;
    int Index2 = 0;
    for (int i=0; i<DIM; i++) {
      Index1 = indices[i];
      for (int j=0; j<i; j++) {
	Index2 = indices[j];
	if ((Index1 == Index2) && (Index1 != -1)) {
	  printf("ARGH, indices[%i] and indices[%i] are both equal to %i\n", i, j, Index1);
	  getchar();
	}
      }
    }
    
    double Displacement = 0.0;
    double RMSD_Expect = 0.0;
    for (int i=0; i<DIM; i++) {
      if (indices[i] != -1) {
	Displacement = sqrtf((double)DistanceMatrix[i*DIM + indices[i]] / conv2);
	RMSD_Expect += Displacement * Displacement;
	//printf("indices[%i] = %i (Disp = % .3f)\n", i, indices[i], Displacement);
      }
    }
    RMSD_Expect = sqrtf(RMSD_Expect / Subset);
    //printf("Expected RMSD from waters is % .3f\n",RMSD_Expect);
    //printf("%i atoms are mapped\n", MapCount);
    //getchar();
  } else {
    apc(DIM,DistanceMatrix,INF,&z_p,indices);
  }

  /*
  //printf("YOOOOO!\n");
  double Displacement;
  double RMSD_Expect;
  for (int i=0; i<Subset; i++) {
    //Element = (double)DistanceMatrix[i*DIM + indices[i]];
    Displacement = sqrtf(Element / conv2);
    RMSD_Expect += Displacement * Displacement;
  }
  RMSD_Expect = sqrtf(RMSD_Expect / Subset);
  //printf("Expected RMSD from waters is % .3f\n",RMSD_Expect);
  */
  time_accumulate(&APCTime,start);
  free(DistanceMatrix);
  if (bRect) {
    free(DistanceMatrixA);
    free(DistanceMatrixB); 
    free(indicesA);
    free(indicesB);
    free(indicesBT);
    free(indicesCount);
    free(FillSort);
    free(FillDist);
    free(FillIdx);
  }
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
  
  /**********************************/
  /*   Initialize input variables   */
  /**********************************/
  // TheoData for distinguishable atoms
  PyArrayObject *XYZ_id_a_, *XYZ_id_b_, *G_id_a_;
  int nreal_id=-1,npad_id=-1,strd_id=-1;
  float G_id_b=-1;
  // Arrays for permutable indices and permutable atom 'batch' size (i.e. oxygens, hydrogens)
  PyArrayObject *LP_Flat_,*LP_Lens_,*LP_Lens_B_;
  // Array for alternate indices
  PyArrayObject *Alt_Idx_;
  // Array for distinguishable indices
  PyArrayObject *Id_Idx_;
  // Arrays for RMSD and rotation matrices
  PyArrayObject *RMSD_, *Rotations_;
  // The entire set of XYZ coordinates for fitting trajectory and reference frame
  PyArrayObject *XYZ_all_a_, *XYZ_all_b_;
  int Usage=-1;
  // Rectangular permutations (EXPERIMENTAL)
  int bRect=0;
  // Debug printout on or off.
  int DebugPrint = 0;
  
  float msd;
  
  if (!PyArg_ParseTuple(args, "iiiiiOOOfOOOOOOOO", &Usage, &DebugPrint,
			&nreal_id, &npad_id, &strd_id, &XYZ_id_a_, &XYZ_id_b_, &G_id_a_, &G_id_b, 
			&Id_Idx_, &LP_Flat_, &LP_Lens_, &LP_Lens_B_, &Alt_Idx_, &Rotations_, &XYZ_all_a_, &XYZ_all_b_)) {
    printf("Mao says: Inputs / outputs not correctly specified!\n");
    return NULL;
  }

  if (DebugPrint) {
    start = get_time_precise();
    printf("Preparing...\n");
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
  // Number of permutable atom groups
  int n_lp_grps = LP_Lens_->dimensions[0];
  // Number of permutable (or alternate) atoms
  int lplen = LP_Flat_->dimensions[0];
  // Number of alternate index atoms
  int altlen = Alt_Idx_->dimensions[0];
  // Number of distinguishable atoms (true)
  int na_id = nreal_id;
  // Number of distinguishable+permutable atoms
  int na_lp = na_id + lplen;
  // TheoData for distinguishable atoms
  float *IDAtoms_Traj_f = (float*) XYZ_id_a_->data;
  float *IDAtoms0_f = (float*) XYZ_id_b_->data;
  float *G_id_a = (float*) G_id_a_->data;
  // Arrays for permutable indices and permutable atom group size
  long unsigned int *lp_lens   = (long unsigned int*) LP_Lens_->data;
  long unsigned int *lp_lens_B = (long unsigned int*) LP_Lens_B_->data;
  long unsigned int *lp_idx   = (long unsigned int*) LP_Flat_->data;
  // Arrays for distinguishable atom indices
  long unsigned int *id_idx = (long unsigned int*) Id_Idx_->data;
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
  float *AllAtoms0_f = (float*) XYZ_all_b_->data;
  
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
    printf("Usage Mode: %i\n",Usage);
    PrintDimensions("Rotations_", Rotations_);
    PrintDimensions("XYZ_all_a_", XYZ_all_a_);
    PrintDimensions("XYZ_all_b_", XYZ_all_b_);
    PrintDimensions("LP_Lens_", LP_Lens_);
    PrintDimensions("LP_Lens_B_", LP_Lens_B_);
    PrintDimensions("LP_Flat_", LP_Flat_);
    PrintDimensions("Alt_Idx_", Alt_Idx_);
    printf("LP_Lens has this many dimensions: %i\n",LP_Lens_->nd);
  }
  
  
  /*********************************/
  /* Initialize internal variables */
  /*   (Memory is allocated here)  */
  /*********************************/
  // Temporary labels for old and new index
  int StartIdx;
  int *lp_starts;
  float *LPAtoms0_f;
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
	  printf("%i ",lp_idx[lp_starts[i] + j]);
	printf("\n");
      }
    }
    LPAtoms0_f = calloc(na_lp * 3, sizeof(double));
    for (int j=0; j<na_id; j++) {
      LPAtoms0_f[0*na_lp + j] = (float) AllAtoms0_f[0*na_all + id_idx[j]];
      LPAtoms0_f[1*na_lp + j] = (float) AllAtoms0_f[1*na_all + id_idx[j]];
      LPAtoms0_f[2*na_lp + j] = (float) AllAtoms0_f[2*na_all + id_idx[j]];
    }
    for (int j=0; j<lplen; j++) {
      LPAtoms0_f[0*na_lp + na_id + j] = (float) AllAtoms0_f[0*na_all + lp_idx[j]];
      LPAtoms0_f[1*na_lp + na_id + j] = (float) AllAtoms0_f[1*na_all + lp_idx[j]];
      LPAtoms0_f[2*na_lp + na_id + j] = (float) AllAtoms0_f[2*na_all + lp_idx[j]];
    }
  }

  if (HaveID) {
    if (DebugPrint) {
      printf("Normal Indices has Length %i \n",na_id);
      for (int i = 0; i<na_id; i++) {
	printf("%i ",id_idx[i]);
      }
      printf("\n");
    }
  }  

  if (HaveAlt) {
    if (DebugPrint) {
      printf("Alternate Indices has Length %i \n",altlen);
      for (int i = 0; i<altlen; i++) {
	printf("%i ",alt_idx[i]);
      }
      printf("\n");
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
  
  /**********************************/
  /*   Start the RMSD calculation   */
  /*    Parallelized over frames    */
  /**********************************/
  // Local working variables
  float x1, x2, y1, y2, z1, z2;
  int Old, New;
  double *MapAtoms0_d;
  double *MapAtoms_d;
  float *MapAtoms0_f;
  float *MapAtoms_f;
  float *MapPadAtoms0_f;
  float *MapPadAtoms_f;
  int Count=0;
  int AtomsInRMSD=0;
  double xcena, ycena, zcena, xcenb, ycenb, zcenb;
  double G_map_d;
  double G_map_f;
  double G_map0_d;
  double G_map0_f;
  double msdi;
  double Displacement;
  int D;
  double *LPAtoms_IDRot_d;
  float  *LPAtoms_IDRot_f;
  double *AllAtoms_d;
  double *AllAtoms_Rot_d;
  double *AllAtoms_Rot_Perm_d;
  double *LPAtoms_d;
  int *LPIndices_Local;
  int *LPIndices_Global;
  double *AltAtoms_d;
  double *AltAtoms_Rot_d;
  int nreal_map = na_lp;
  int npad_map = na_lp+4-na_lp%4;
#pragma omp parallel for private(rot, msd, msdi, j, k, p, start, end, Idx, Old, New, Count, x1, x2, y1, y2, z1, z2, MapAtoms0_d, MapAtoms_d, MapAtoms0_f, MapAtoms_f, MapPadAtoms0_f, MapPadAtoms_f, G_map_d, G_map0_d, G_map_f, G_map0_f, AtomsInRMSD, xcena, ycena, zcena, xcenb, ycenb, zcenb, Displacement, D, LPAtoms_d, LPAtoms_IDRot_d, LPAtoms_IDRot_f, AllAtoms_d, AllAtoms_Rot_d, AllAtoms_Rot_Perm_d, LPIndices_Local, LPIndices_Global, AltAtoms_d, AltAtoms_Rot_d)
  for (int i = 0; i < ns; i++) 
    {
      MapAtoms0_d = calloc(na_lp * 3, sizeof(double));
      MapAtoms0_f = calloc(na_lp * 3, sizeof(float));
      MapAtoms_d = calloc(na_lp * 3, sizeof(double));
      MapAtoms_f = calloc(na_lp * 3, sizeof(float));
      MapPadAtoms0_f = calloc(npad_map * 3, sizeof(float));
      MapPadAtoms_f = calloc(npad_map * 3, sizeof(float));
      
      LPAtoms_d = calloc(na_lp * 3, sizeof(double));
      LPAtoms_IDRot_d = calloc(na_lp * 3, sizeof(double));
      LPAtoms_IDRot_f = calloc(na_lp * 3, sizeof(float));
      for (int j=0; j<na_id; j++) {
	LPAtoms_d[0*na_lp + j] = (double) *(XYZ_all_a + i*na_all*3 + 0*na_all + id_idx[j]);
	LPAtoms_d[1*na_lp + j] = (double) *(XYZ_all_a + i*na_all*3 + 1*na_all + id_idx[j]);
	LPAtoms_d[2*na_lp + j] = (double) *(XYZ_all_a + i*na_all*3 + 2*na_all + id_idx[j]);
      }
      for (int j=0; j<lplen; j++) {
	LPAtoms_d[0*na_lp + na_id + j] = (double) *(XYZ_all_a + i*na_all*3 + 0*na_all + lp_idx[j]);
	LPAtoms_d[1*na_lp + na_id + j] = (double) *(XYZ_all_a + i*na_all*3 + 1*na_all + lp_idx[j]);
	LPAtoms_d[2*na_lp + na_id + j] = (double) *(XYZ_all_a + i*na_all*3 + 2*na_all + lp_idx[j]);
      }

      AllAtoms_d = calloc(na_all*3, sizeof(double));
      AllAtoms_Rot_d = calloc(na_all*3, sizeof(double));
      AllAtoms_Rot_Perm_d = calloc(na_all*3, sizeof(double));
      for (int j=0; j<na_all*3; j++) {
	AllAtoms_d[j] = (double) XYZ_all_a[na_all*3*i+j];
      }
      
      LPIndices_Local = calloc(lplen, sizeof(int));
      LPIndices_Global = calloc(lplen, sizeof(int));

      AltAtoms_d = calloc(altlen * 3, sizeof(double));
      AltAtoms_Rot_d = calloc(altlen * 3, sizeof(double));
      for (int j=0; j < altlen ; j++) {
	AltAtoms_d[0*altlen + j] = (double) *(XYZ_all_a + i*na_all*3 + 0*na_all + alt_idx[j]);
	AltAtoms_d[1*altlen + j] = (double) *(XYZ_all_a + i*na_all*3 + 1*na_all + alt_idx[j]);
	AltAtoms_d[2*altlen + j] = (double) *(XYZ_all_a + i*na_all*3 + 2*na_all + alt_idx[j]);
      }
      //If there are AtomIndices:
      if (HaveID) {
	start = get_time_precise();
	// Perform alignment using AtomIndices; this sets the RMSD values and optionally gets the rotation matrices.
	ls_rmsd2_aligned_T_g(nreal_id,npad_id,strd_id,(IDAtoms_Traj_f+i*3*npad_id),IDAtoms0_f,G_id_a[i],G_id_b,&msd,(HaveLP || HaveAlt || WantXYZ),rot);
	time_accumulate(&RMSD1Time,start);
	// If there are no PermuteIndices:
	if (!HaveLP) {
	  start = get_time_precise();
	  // If we want output coordinates:
	  if (WantXYZ) {
	    // Rotate the whole frame using the rotation matrix
	    cblas_dgemm(101,112,111,3,na_all,3,1.0,rot,3,AllAtoms_d,na_all,0.0,AllAtoms_Rot_d,na_all);
	    time_accumulate(&MatrixTime,start);
	  }
	  // If there are AltIndices:
	  if (HaveAlt) {
	    // Rotate the atoms in AltIndices using the rotation matrix.
	    cblas_dgemm(101,112,111,3,altlen,3,1.0,rot,3,AltAtoms_d,altlen,0.0,AltAtoms_Rot_d,altlen);
	    time_accumulate(&MatrixTime,start);
	    start = get_time_precise();
	    msd = 0.0 ;
	    // Set the RMSD values by explicitly computing them from pairwise distances.
	    for (j=0; j<altlen; j++) {
	      Idx = alt_idx[j];
	      x2 = AltAtoms_Rot_d[0*altlen + j];
	      y2 = AltAtoms_Rot_d[1*altlen + j];
	      z2 = AltAtoms_Rot_d[2*altlen + j];
	      x1 = AllAtoms0_f[0*na_all + Idx];
	      y1 = AllAtoms0_f[1*na_all + Idx];
	      z1 = AllAtoms0_f[2*na_all + Idx];
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
	  cblas_dgemm(101,112,111,3,na_lp,3,1.0,rot,3,LPAtoms_d,na_lp,0.0,LPAtoms_IDRot_d,na_lp);
	}
      }
      // If there are PermuteIndices:
      if (HaveLP) {
	time_accumulate(&MatrixTime,start);
	for (k=0; k<na_lp * 3 ; k++) {
	  LPAtoms_IDRot_f[k] = (float) LPAtoms_IDRot_d[k];
	}
	// Permute the atomic indices for each batch in PermuteIndices.
	Count = 0;
	for (k=0; k<na_id; k++) {
	  MapAtoms_d[0*na_lp + Count] = (double)LPAtoms_d[0*na_lp + Count];
	  MapAtoms_d[1*na_lp + Count] = (double)LPAtoms_d[1*na_lp + Count];
	  MapAtoms_d[2*na_lp + Count] = (double)LPAtoms_d[2*na_lp + Count];
	  MapAtoms0_d[0*na_lp + Count] = (double)LPAtoms0_f[0*na_lp + Count];
	  MapAtoms0_d[1*na_lp + Count] = (double)LPAtoms0_f[1*na_lp + Count];
	  MapAtoms0_d[2*na_lp + Count] = (double)LPAtoms0_f[2*na_lp + Count];
	  Count++;
	}

	for (k=0; k<n_lp_grps ; k++) {
	  start = get_time_precise();
	  // This calls a subroutine that builds the cost matrix and solves the assignment problem!
	  drive_permute(lp_lens[k],na_lp,na_id+lp_starts[k],LPAtoms_IDRot_f,LPAtoms0_f,LPIndices_Local+lp_starts[k],lp_lens_B[k]);
	  //printf("YOU BOYS LIKE MEXICO??\n");
	  time_accumulate(&PermuteTime,start);
	  start = get_time_precise();
	  // Relabel the atoms according to our brand new permutations
	  for (p=0; p<lp_lens[k] ; p++) {
	    if (*(LPIndices_Local+lp_starts[k]+p) >= 0) {
	      Old = na_id + lp_starts[k] + p;
	      New = na_id + lp_starts[k] + *(LPIndices_Local+lp_starts[k]+p) ;

	      MapAtoms_d[0*na_lp + Count] = (double)LPAtoms_d[0*na_lp + New];
	      MapAtoms_d[1*na_lp + Count] = (double)LPAtoms_d[1*na_lp + New];
	      MapAtoms_d[2*na_lp + Count] = (double)LPAtoms_d[2*na_lp + New];
	      MapAtoms0_d[0*na_lp + Count] = (double)LPAtoms0_f[0*na_lp + Old];
	      MapAtoms0_d[1*na_lp + Count] = (double)LPAtoms0_f[1*na_lp + Old];
	      MapAtoms0_d[2*na_lp + Count] = (double)LPAtoms0_f[2*na_lp + Old];

	      Count++;
	      New = lp_idx[lp_starts[k] + *(LPIndices_Local+lp_starts[k]+p)];
	      LPIndices_Global[lp_starts[k]+p] = New;
	    } else {
	      LPIndices_Global[lp_starts[k]+p] = -1;
	    }

	  }
	  time_accumulate(&RelabelTime,start);
	}

	// This block computes the center of mass and the Theobald G-value.
	// It looks like previously I had been trying to determine the sizes of the "mapatoms" dynamically.
	// This was probably because of the "boundary problem".
	// Disabling this so that we recover the original functionality.
	//nreal_map = Count;
	//npad_map = Count+4-Count%4;
	xcena = 0.0;
	ycena = 0.0;
	zcena = 0.0;
	xcenb = 0.0;
	ycenb = 0.0;
	zcenb = 0.0;
	G_map_d = 0.0;
	G_map0_d = 0.0;

	for (k=0; k<Count; k++) {
	  xcena += MapAtoms_d[0*na_lp + k];
	  ycena += MapAtoms_d[1*na_lp + k];
	  zcena += MapAtoms_d[2*na_lp + k];

	  xcenb += MapAtoms0_d[0*na_lp + k];
	  ycenb += MapAtoms0_d[1*na_lp + k];
	  zcenb += MapAtoms0_d[2*na_lp + k];
	}
	xcena /= Count;
	ycena /= Count;
	zcena /= Count;
	xcenb /= Count;
	ycenb /= Count;
	zcenb /= Count;
	for (k=0; k<Count; k++) {
	  MapAtoms_d[0*na_lp + k] -= xcena;
	  MapAtoms_d[1*na_lp + k] -= ycena;
	  MapAtoms_d[2*na_lp + k] -= zcena;

	  MapAtoms0_d[0*na_lp + k] -= xcenb;
	  MapAtoms0_d[1*na_lp + k] -= ycenb;
	  MapAtoms0_d[2*na_lp + k] -= zcenb;

	  G_map_d += MapAtoms_d[0*na_lp + k]*MapAtoms_d[0*na_lp + k];
	  G_map_d += MapAtoms_d[1*na_lp + k]*MapAtoms_d[1*na_lp + k];
	  G_map_d += MapAtoms_d[2*na_lp + k]*MapAtoms_d[2*na_lp + k];
	  G_map0_d += MapAtoms0_d[0*na_lp + k]*MapAtoms0_d[0*na_lp + k];
	  G_map0_d += MapAtoms0_d[1*na_lp + k]*MapAtoms0_d[1*na_lp + k];
	  G_map0_d += MapAtoms0_d[2*na_lp + k]*MapAtoms0_d[2*na_lp + k];
	}
	G_map_f = (float) G_map_d;
	G_map0_f = (float) G_map0_d;

	for (k=0; k<Count; k++) {
	  MapAtoms_f[0*na_lp + k] = (float)MapAtoms_d[0*na_lp + k];
	  MapAtoms_f[1*na_lp + k] = (float)MapAtoms_d[1*na_lp + k];
	  MapAtoms_f[2*na_lp + k] = (float)MapAtoms_d[2*na_lp + k];
	  MapAtoms0_f[0*na_lp + k] = (float)MapAtoms0_d[0*na_lp + k];
	  MapAtoms0_f[1*na_lp + k] = (float)MapAtoms0_d[1*na_lp + k];
	  MapAtoms0_f[2*na_lp + k] = (float)MapAtoms0_d[2*na_lp + k];
	}

	// Compress the arrays so they fit in npad_map.
	for (D=0; D<3; D++) {
	  for (k=0; k<nreal_map; k++) {
	    MapPadAtoms_f[D*npad_map + k] = MapAtoms_f[D*na_lp + k];
	    MapPadAtoms0_f[D*npad_map + k] = MapAtoms0_f[D*na_lp + k];
	  }
	  for (k=nreal_map; k<na_lp; k++) {
	    MapPadAtoms_f[D*npad_map + k] = 0.0;
	    MapPadAtoms0_f[D*npad_map + k] = 0.0;
	  }
	}
	start = get_time_precise();
	// Perform alignment using the Mapped Indices; this sets the RMSD values and gets the rotation matrices
	ls_rmsd2_aligned_T_g(nreal_map,npad_map,npad_map,MapPadAtoms_f,MapPadAtoms0_f,G_map_f,G_map0_f,&msd,1,rot);
	time_accumulate(&RMSD2Time,start);
	
	// Rotate the whole frame using the rotation matrix
	start = get_time_precise();

	cblas_dgemm(101,112,111,3,na_all,3,1.0,rot,3,AllAtoms_d,na_all,0.0,AllAtoms_Rot_d,na_all);
	time_accumulate(&MatrixTime,start);
	start = get_time_precise();
	for (k=0; k<3*na_all ; k++) {
	  *(AllAtoms_Rot_Perm_d+k) = *(AllAtoms_Rot_d+k);
	}
	// Relabel the frame using the permutations
	for (k=0; k<lplen; k++) {
	  Idx = lp_idx[k];
	  New = LPIndices_Global[k];
	  if (New != -1) {
	    //printf("Atom number %i is being relabeled to %i\n",Idx,New);
	    // AllAtoms_Rot_Perm_d contains all of the atoms being fitted, with relabeled atom indices
	    AllAtoms_Rot_Perm_d[0*na_all + Idx] = AllAtoms_Rot_d[0*na_all + New];
	    AllAtoms_Rot_Perm_d[1*na_all + Idx] = AllAtoms_Rot_d[1*na_all + New];
	    AllAtoms_Rot_Perm_d[2*na_all + Idx] = AllAtoms_Rot_d[2*na_all + New];
	    x2 = AllAtoms_Rot_Perm_d[0*na_all + Idx];
	    y2 = AllAtoms_Rot_Perm_d[1*na_all + Idx];
	    z2 = AllAtoms_Rot_Perm_d[2*na_all + Idx];
	    x1 = AllAtoms0_f[0*na_all + Idx];
	    y1 = AllAtoms0_f[1*na_all + Idx];
	    z1 = AllAtoms0_f[2*na_all + Idx];
	    msdi = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
	    //printf("Displacement from reference atom %i to permuted atom %i is % .4f\n", Idx, New, sqrtf(msdi));
	  }
	}
	time_accumulate(&RelabelTime,start);
	start = get_time_precise();
	if (HaveAlt) {
	  AtomsInRMSD = 0;
	  for (j=0; j<altlen; j++) {
	    Idx = alt_idx[j];
	    x2 = AllAtoms_Rot_d[0*na_all + Idx];
	    y2 = AllAtoms_Rot_d[1*na_all + Idx];
	    z2 = AllAtoms_Rot_d[2*na_all + Idx];
	    x1 = AllAtoms0_f[0*na_all + Idx];
	    y1 = AllAtoms0_f[1*na_all + Idx];
	    z1 = AllAtoms0_f[2*na_all + Idx];
	    msdi = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
	    msd += msdi;
	    AtomsInRMSD++;
	  }
	  for (k=0; k<lplen; k++) {
	    Idx = lp_idx[k];
	    New = LPIndices_Global[i*lplen + k];
	    if (New != -1) {
	      x2 = AllAtoms_Rot_Perm_d[0*na_all + Idx];
	      y2 = AllAtoms_Rot_Perm_d[1*na_all + Idx];
	      z2 = AllAtoms_Rot_Perm_d[2*na_all + Idx];
	      x1 = AllAtoms0_f[0*na_all + Idx];
	      y1 = AllAtoms0_f[1*na_all + Idx];
	      z1 = AllAtoms0_f[2*na_all + Idx];
	      msdi = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
	      msd += msdi;
	      AtomsInRMSD++;
	    }
	  }
	  msd /= AtomsInRMSD;
	}
	time_accumulate(&AltRMSDTime,start);
      }
      // Assign the RMSD values to the RMSDOut array
      RMSD[i] = sqrtf(msd);
      //printf("The RMSD value is % .3f\n", RMSD[i]);
      // Assign the rotation matrix to the RotOut array
      for (j=0; j<9; j++) {
	*(Rotations + i*9 + j) = (float) rot[j];
      }
      // If output coordinates are requested:
      // Assign the rotated frames to the XYZOut array
      if (WantXYZ) {
	for (j=0; j<3*na_all; j++) {
	  if (HaveLP)
	    XYZ_all_a[3*na_all*i + j] = AllAtoms_Rot_Perm_d[j];
	  else
	    XYZ_all_a[3*na_all*i + j] = AllAtoms_Rot_d[j];
	}
      }

      free(MapPadAtoms0_f);
      free(MapPadAtoms_f);
      free(MapAtoms0_d);
      free(MapAtoms0_f);
      free(MapAtoms_d);
      free(MapAtoms_f);
      free(LPAtoms_d);
      free(LPAtoms_IDRot_d);
      free(LPAtoms_IDRot_f);
      free(AllAtoms_d);
      free(AllAtoms_Rot_d);
      free(AllAtoms_Rot_Perm_d);
      free(LPIndices_Local);
      free(LPIndices_Global);
      free(AltAtoms_d);
      free(AltAtoms_Rot_d);
      // Pause here after one RMSD (if developing)
      //getchar();
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
    float MaxRMSD = 0.0, MinRMSD = 10.0;
    for (int i = 0; i < ns ; i ++) {
      if (RMSD[i] > MaxRMSD)
	MaxRMSD = RMSD[i];
      if (RMSD[i] < MinRMSD)
	MinRMSD = RMSD[i];
    }
    printf("Min / Max RMSD = % .4f / % .4f\n",MinRMSD, MaxRMSD);
  }
  
  if (HaveLP) {
    free(lp_starts);
    free(LPAtoms0_f);
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
