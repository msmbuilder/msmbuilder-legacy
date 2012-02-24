#include <math.h>
#include "Python.h"
#include "dihedral.h"
#include <numpy/arrayobject.h>
#include <stdio.h>

extern PyObject *dihedrals_from_traj_wrap(PyObject *self, PyObject *args) {
    PyArrayObject *results_, *xyzlist_, *quartets_;
    int traj_length, num_quartets, four, num_atoms, num_dims;
    double *results;
    const double *xyzlist;
    const long *quartets;
    if (!PyArg_ParseTuple(args, "O!O!O!",
            &PyArray_Type, &results_, &PyArray_Type, &xyzlist_, &PyArray_Type, &quartets_)) {
        return 0;
    }
    else {
        results = (double*) results_->data;
        xyzlist = (const double*) xyzlist_->data;
        quartets = (const long*) quartets_->data;
        
        traj_length = xyzlist_->dimensions[0];
        num_atoms = xyzlist_->dimensions[1];
        num_dims = xyzlist_->dimensions[2];
        num_quartets = quartets_->dimensions[0];
        four = quartets_->dimensions[1];
        
        if ((num_dims != 3) || (four != 4)) {
            printf("Incorrect call to dihedrals_from_trajectory_wrap! Aborting");
            exit(1);
        }
        
        dihedrals_from_traj(results, xyzlist, quartets, traj_length, num_atoms, num_quartets);
    }
    return Py_BuildValue("d", 0.0);
}

extern PyObject *dihedrals_from_traj_float_wrap(PyObject *self, PyObject *args) {
    PyArrayObject *results_, *xyzlist_, *quartets_;
    int traj_length, num_quartets, four, num_atoms, num_dims;
    float *results;
    const float *xyzlist;
    const long *quartets;
    if (!PyArg_ParseTuple(args, "O!O!O!",
            &PyArray_Type, &results_, &PyArray_Type, &xyzlist_, &PyArray_Type, &quartets_)) {
        return 0;
    }
    else {
        results = (float*) results_->data;
        xyzlist = (const float*) xyzlist_->data;
        quartets = (const long*) quartets_->data;
        
        traj_length = xyzlist_->dimensions[0];
        num_atoms = xyzlist_->dimensions[1];
        num_dims = xyzlist_->dimensions[2];
        num_quartets = quartets_->dimensions[0];
        four = quartets_->dimensions[1];
        
        if ((num_dims != 3) || (four != 4)) {
            printf("Incorrect call to dihedrals_from_trajectory_wrap! Aborting");
            exit(1);
        }
        
        dihedrals_from_traj_float(results, xyzlist, quartets, traj_length, num_atoms, num_quartets);
    }
    return Py_BuildValue("d", 0.0);
}

static PyMethodDef _dihedralWrapMethods[] = {
  {"dihedrals_from_traj_wrap", dihedrals_from_traj_wrap, METH_VARARGS},
  {"dihedrals_from_traj_float_wrap", dihedrals_from_traj_float_wrap, METH_VARARGS},
  {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

DL_EXPORT(void) init_dihedral_wrap(void) {
  Py_InitModule3("_dihedral_wrap", _dihedralWrapMethods, "Wrappers for dihedral calculation.");
  import_array();
}

