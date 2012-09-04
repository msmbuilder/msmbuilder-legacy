#include <math.h>
#include "Python.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include "asa.h"

// python call signigute: _asa.asa_frame(frame, atom_radii, n_sphere_points, areas)
// Compute the accessible surface area of each atom in an MD snapshot
// Parameters
// ----------
// frame : np.ndarray, shape=[n_atoms, 3], dtype=np.float32
//     The xyz coordinates
// atom_radii : np.ndarray, shape=[n_atoms], dtype=np.float32
//     The radii of the atoms PLUS the probe radius
// n_sphere_points : int,
//     number of points to generate sampling the unit sphere. higher is
//     better (more accurate) but more expensive
// areas : np.ndarray, shape=[n_atoms], dtype=np.float32
//     Results will be put here
extern PyObject *_asa_frame(PyObject *self, PyObject *args) {
    PyArrayObject *frame_, *atom_radii_, *areas_;
    const float *frame, *atom_radii;
    float *areas;
    int n_atoms, n_sphere_points;

    if (!PyArg_ParseTuple(args, "O!O!iO!",
            &PyArray_Type, &frame_,
            &PyArray_Type, &atom_radii_,
            &n_sphere_points,
            &PyArray_Type, &areas_)) {
        return 0;
    }
    else {
        frame = (const float*)frame_->data;
        n_atoms = frame_->dimensions[0];
        if (frame_->dimensions[1] != 3 || frame_->nd != 2 ||
            frame_->descr->elsize != sizeof(float)) {
            PyErr_SetString(PyExc_TypeError, "frame");
            return 0;
        }
        
        atom_radii = (const float*)atom_radii_->data;
        if (atom_radii_->nd != 1 ||
            atom_radii_->descr->elsize != sizeof(float)) {
            PyErr_SetString(PyExc_TypeError, "radii");                
            return 0;
        }
        
        areas = (float*)areas_->data;
        if (areas_->dimensions[0] != n_atoms ||
            areas_->descr->elsize != sizeof(float)) {
            PyErr_SetString(PyExc_TypeError, "areas");                
             return 0;
        }
        
        // just call asa_trajectory saying we have only 1 frame
        asa_trajectory(frame, 1, n_atoms, atom_radii, n_sphere_points, areas);
        
    }
    return Py_BuildValue("d", 0.0);
}


// python call signigute: _asa.asa_traj(xyzlist, atom_radii, n_sphere_points, results)
// Compute the accessible surface area of every snapshot in an MD trajectory
// Parameters
// ----------
// xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3], dtype=np.float32
//     The xyz coordinates of th trajectory
// atom_radii : np.ndarray, shape=[n_atoms], dtype=np.float32
//     The radii of the atoms PLUS the probe radius
// n_sphere_points : int,
//     number of points to generate sampling the unit sphere. higher is
//     better (more accurate) but more expensive
// results : np.ndarray, shape=[n_frames, n_atoms], dtype=np.float32
//     Results will be put here
extern PyObject *_asa_traj(PyObject *self, PyObject *args) {
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


/*
Biolerplate to make these functions available in python
*/
static PyMethodDef _asa_methods[] = {
  {"asa_traj", _asa_traj, METH_VARARGS},
  {"asa_frame", _asa_frame, METH_VARARGS},
  {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_asa",
    NULL,
    -1,
    _asa_methods,
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
  (void) Py_InitModule("_asa", _asa_methods);
  import_array();  // Must be present for NumPy.  Called first after above line.
}
#endif