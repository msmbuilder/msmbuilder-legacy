# methods to support testing
import os
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
  assert_approx_equal, assert_array_almost_equal, assert_array_almost_equal_nulp,
  assert_array_equal, assert_array_less, assert_array_max_ulp, assert_equal,
  assert_raises, assert_string_equal, assert_warns)
from nose.tools import ok_, eq_

from pkg_resources import resource_filename

__all__ = ['get', 'eq' 'assert_traj_equal', 'assert_spase_matrix_equal',
           # stuff that was imported from numpy / nose too
          'ok_', 'eq_', 'assert_allclose', 'assert_almost_equal',
          'assert_approx_equal', 'assert_array_almost_equal',
          'assert_array_almost_equal_nulp', 'assert_array_equal',
          'assert_array_less', 'assert_array_max_ulp', 'assert_equal',
          'assert_raises', 'assert_string_equal', 'assert_warns']

def get(self, name, just_filename=False):
    """"
    Get reference data for testing

    Parameters
    ----------
    name : string
        name of the resource, ususally a filename or path
    just_filename : bool, optional
        if true, we just return the filename. otherwise we try to load
        up the data from disk

    Notes
    -----
    The heuristic basically just looks at the filename extension, but it
    has a few tricks, like loading AtomIndices ith dtype=np.int

    """
    # delay these imports, since this module is loaded in a bunch
    # of places but not necessarily used
    import scipy.io
    from msmbuilder import Trajectory, Serializer

    # ReferenceData is where all the data is stored
    fn = resource_filename('msmbuilder', os.path.join('ReferenceData', name))

    if just_filename:
        return fn

    # the filename extension
    ext = os.path.splitext(fn)[1]

    # load trajectories
    if ext in ['.lh5', '.pdb']:
        val = Trajectory.LoadTrajectoryFile(fn)

    # load flat text files
    elif 'AtomIndices.dat' in fn:
        # try loading AtomIndices first, because the default for loadtxt
        # is to use floats
        val = np.loadtxt(fn, dtype=np.int)
    elif ext in ['.dat']:
        # try loading general .dats with floats
        val = np.loadtxt(fn)

    # load with serializer files that end with .h5, .hdf or .h5.distances
    elif ext in ['.h5', '.hdf']:
        val = Serializer.LoadFromHDF(fn)
    elif fn.endswith('.h5.distances'):
        val = Serializer.LoadFromHDF(fn)

    # load matricies
    elif ext in ['.mtx']:

        val = scipy.io.mmread(fn)
    else:
        raise TypeError("I could not infer how to load this file. You "
            "can either request load=False, or perhaps add more logic to "
            "the load heuristics in this class: %s" % fn)

    return val


def eq(o1, o2, decimal=6):
    from msmbuilder import Trajectory
    from scipy.sparse import isspmatrix

    assert (type(o1) is type(o2)), 'o1 and o2 not the same type: %s %s' % (type(o1), type(o2))

    if isinstance(o1, Trajectory):
        assert_traj_equal(o1, o1, decimal)
    elif isspmatrix(o1):
        assert_spase_matrix_equal(o1, o1, decimal)
    elif isinstance(o1, np.ndarray)
        if o1.dtype.kind == 'f' or o2.dtype.kind == 'f':
            # compare floats for almost equality
            assert_array_almost_equal(o1, o2, decimal)
        else:
            # compare everything else (ints, bools) for absolute equality
            assert_array_equal(val, t2[key])
    # probably these are other specialized types
    # that need a special check?
    else:
        eq_(o1, o2)


def assert_traj_equal(t1, t2, decimal=6):
    """Assert two msmbuilder trajectories are equal. This method should actually
    work for any dict of numpy arrays/objects """

    # make sure the keys are the same
    eq_(t1.keys(), t2.keys())

    for key, val in t1.iteritems():
        # compare numpy arrays using numpy.testing
        if isinstance(val, np.ndarray):
            if val.dtype.kind ==  'f':
                # compare floats for almost equality
                assert_array_almost_equal(val, t2[key], decimal)
            else:
                # compare everything else (ints, bools) for absolute equality
                assert_array_equal(val, t2[key])
        else:
            eq_(val == t2[key])

def assert_spase_matrix_equal(m1, m2, decimal=6):
    """Assert two scipy.sparse matrices are equal."""

    # delay the import to speed up stuff if this method is unused
    from scipy.sparse import isspmatrix
    from numpy.linalg import norm

    # both are sparse matricies
    assert isspmatrix(m1)
    assert isspmatrix(m1)

    # make sure they have the same format
    eq_(m1.format, m2.format)

    # even though its called assert_array_almost_equal, it will
    # work for scalars
    assert_array_almost_equal(norm(m1 - m2), 0, decimal=decimal)
