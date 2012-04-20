#include <math.h>
#include "Python.h"
#include "rg.h"
#include <numpy/arrayobject.h>
#include <stdio.h>

extern PyObject *rg_wrap(PyObject *self, PyObject *args) {
    PyArrayObject *results_, *xyzlist_;
    int traj_length, num_atoms, num_dims;
    double *results;
    const float *xyzlist;
    if (!PyArg_ParseTuple(args, "O!O!",
            &PyArray_Type, &xyzlist_, &PyArray_Type, &results_)) {
        return 0;
    }
    else {
        results = (double*) results_->data;
        xyzlist = (const float*) xyzlist_->data;
        
        traj_length = xyzlist_->dimensions[0];
        num_atoms = xyzlist_->dimensions[1];
        num_dims = xyzlist_->dimensions[2];
        
        if (num_dims != 3) {
            printf("Incorrect call to rg_wrap! Aborting");
            exit(1);
        }
        rg(xyzlist, traj_length, num_atoms, results);
    }
    return Py_BuildValue("d", 0.0);
}

static PyMethodDef _rgWrapMethods[] = {
  {"rg_wrap", rg_wrap, METH_VARARGS},
  {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

DL_EXPORT(void) init_rg_wrap(void) {
  Py_InitModule3("_rg_wrap", _rgWrapMethods, "Wraper for rg calculation.");
  import_array();
}

        