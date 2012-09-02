#include <math.h>
#include "Python.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include "asa.h"

// python call signigute: asa_traj_wrap(xyzlist, atom_radii, n_sphere_points, results)
extern PyObject *asa_traj_wrap(PyObject *self, PyObject *args) {
    PyArrayObject *xyzlist_, *atom_radii_, *results_;
    const float *xyzlist, *atom_radii;
    float *results;
    int n_frames, n_atoms, n_sphere_points;

    if (!PyArg_ParseTuple(args, "O!O!iO!",
            &PyArray_Type, &xyzlist_,
            &PyArray_Type, &atom_radii_,
            &n_sphere_points,
            &PyArray_Type, &results_)) {
        return 0;
    }
    else {
        xyzlist = (const float*)xyzlist_->data;
        n_frames = xyzlist_->dimensions[0];
        n_atoms = xyzlist_->dimensions[1];
        if (xyzlist_->dimensions[2] != 3 ||
            xyzlist_->nd != 3 ||
            xyzlist_->descr->elsize != sizeof(float)) {
            PyErr_SetString(PyExc_TypeError, "xyzlist");
            return 0;
        }
        
        atom_radii = (const float*)atom_radii_->data;
        if (atom_radii_->nd != 1 ||
            atom_radii_->descr->elsize != sizeof(float)) {
            PyErr_SetString(PyExc_TypeError, "atom_radii");
            return 0;
        }
        
        results = (float*)results_->data;
        if (results_->dimensions[0] != n_frames ||
            results_->dimensions[1] != n_atoms ||
            results_->descr->elsize != sizeof(float)) {
             PyErr_SetString(PyExc_TypeError, "results");
             return 0;
        }
        
        asa_trajectory(xyzlist, n_frames, n_atoms, atom_radii, n_sphere_points, results);
        
    }
    return Py_BuildValue("d", 0.0);
}



static PyMethodDef _asa_frame_methods[] = {
  {"asa_traj_wrap", asa_traj_wrap, METH_VARARGS},
  {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_asa",
    NULL,
    -1,
    _asa_frame_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit__asa(void)
{
    PyObject *m;

    m = PyModule_Create(&moduledef);
    import_array();

    return m;
}
#else
PyMODINIT_FUNC init_asa(void)
{
  (void) Py_InitModule("_asa", _asa_frame_methods);
  import_array();  // Must be present for NumPy.  Called first after above line.
}
#endif