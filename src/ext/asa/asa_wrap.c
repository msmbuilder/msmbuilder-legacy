#include <math.h>
#include "Python.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include "asa.h"

// python call signigute: asa_frame_wrap(frame, atom_radii, sphere_points, areas)
extern PyObject *asa_frame_wrap(PyObject *self, PyObject *args) {
    PyArrayObject *frame_, *atom_radii_, *sphere_points_, *areas_;
    const float *frame, *atom_radii, *sphere_points;
    float *areas;
    int n_atoms, n_sphere_points;

    if (!PyArg_ParseTuple(args, "O!O!O!O!",
            &PyArray_Type, &frame_,
            &PyArray_Type, &atom_radii_,
            &PyArray_Type, &sphere_points_, 
            &PyArray_Type, &areas_)) {
        return 0;
    }
    else {
        frame = (const float*)frame_->data;
        n_atoms = frame_->dimensions[0];
        if (frame_->dimensions[1] != 3 || frame_->nd != 2 ||
            frame_->descr->elsize != sizeof(float)) {
            return 0;
        }
        
        atom_radii = (const float*)atom_radii_->data;
        if (atom_radii_->nd != 1 ||
            atom_radii_->descr->elsize != sizeof(float)) {
            return 0;
        }
        
        sphere_points = (const float*)sphere_points_->data;
        n_sphere_points = sphere_points_->dimensions[0];
        if (sphere_points_->dimensions[1] != 3 ||
            sphere_points_->nd != 2 || 
            sphere_points_->descr->elsize != sizeof(float)) {
            return 0;
        }
        
        areas = (float*)areas_->data;
        if (areas_->dimensions[0] != n_atoms ||
            areas_->descr->elsize != sizeof(float)) {
             return 0;
        }
        
        asa_frame(frame, n_atoms, atom_radii, sphere_points, n_sphere_points, areas);
        
    }
    return Py_BuildValue("d", 0.0);
}



static PyMethodDef _asa_frame_methods[] = {
  {"asa_frame", asa_frame, METH_VARARGS},
  {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_asa_wrap",
    NULL,
    -1,
    _asa_frame_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit__asa_wrap(void)
{
    PyObject *m;

    m = PyModule_Create(&moduledef);
    import_array();

    return m;
}
#else
PyMODINIT_FUNC init_asa_wrap(void)
{
  (void) Py_InitModule("_asa_wrap", _asa_frame_methods);
  import_array();  // Must be present for NumPy.  Called first after above line.
}
#endif