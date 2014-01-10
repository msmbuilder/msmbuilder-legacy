# methods to support testing
import os
import re
import functools
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
  assert_approx_equal, assert_array_almost_equal, assert_array_almost_equal_nulp,
  assert_array_equal, assert_array_less, assert_array_max_ulp, assert_equal,
  assert_raises, assert_string_equal, assert_warns)
from nose.tools import ok_, eq_, raises
from nose import SkipTest
import mdtraj as md
from mdtraj import io

from pkg_resources import resource_filename

__all__ = ['get', 'load', 'eq', 'assert_dict_equal', 'assert_sparse_matrix_equal',
           'expected_failure', 'skip',
           # stuff that was imported from numpy / nose too
          'ok_', 'eq_', 'assert_allclose', 'assert_almost_equal',
          'assert_approx_equal', 'assert_array_almost_equal',
          'assert_array_almost_equal_nulp', 'assert_array_equal',
          'assert_array_less', 'assert_array_max_ulp', 'assert_equal',
          'assert_raises', 'assert_string_equal', 'assert_warns', 'raises']

def get(name, just_filename=False):
    """
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

    # reference is where all the data is stored
    fn = resource_filename('msmbuilder', os.path.join('reference', name))
    
    if not os.path.exists(fn):
        raise ValueError('Sorry! %s does not exists. If you just '
            'added it, you\'ll have to re install' % fn)

    if just_filename:
        return fn
    return load(fn)
    
def load(filename):
    # delay these imports, since this module is loaded in a bunch
    # of places but not necessarily used
    import scipy.io
    from msmbuilder import Project
    
    # the filename extension
    ext = os.path.splitext(filename)[1]

    # load trajectories
    if ext != '.h5' and ext in md._FormatRegistry.loaders.keys():
        val = md.load(filename)

    # load flat text files
    elif 'AtomIndices.dat' in filename:
        # try loading AtomIndices first, because the default for loadtxt
        # is to use floats
        val = np.loadtxt(filename, dtype=np.int)
    elif ext in ['.dat']:
        # try loading general .dats with floats
        val = np.loadtxt(filename)
    
    # short circuit opening ProjectInfo
    elif ('ProjectInfo.yaml' in filename) or ('ProjectInfo.h5' in filename) or (re.search('ProjectInfo.*\.yaml', filename)):
        val = Project.load_from(filename)
        
    # load with serializer files that end with .h5, .hdf or .h5.distances
    elif ext in ['.h5', '.hdf']:
        val = io.loadh(filename, deferred=False)
    elif filename.endswith('.h5.distances'):
        val = io.loadh(filename, deferred=False)

    # load matricies
    elif ext in ['.mtx']:
        val = scipy.io.mmread(filename)
        
    else:
        raise TypeError("I could not infer how to load this file. You "
            "can either request load=False, or perhaps add more logic to "
            "the load heuristics in this class: %s" % filename)

    return val


def eq(o1, o2, decimal=6):
    from scipy.sparse import isspmatrix

    assert (type(o1) is type(o2)), 'o1 and o2 not the same type: %s %s' % (type(o1), type(o2))

    if isinstance(o1, dict):
        assert_dict_equal(o1, o2, decimal)
    elif isspmatrix(o1):
        assert_sparse_matrix_equal(o1, o2, decimal)
    elif isinstance(o1, np.ndarray):
        if o1.dtype.kind == 'f' or o2.dtype.kind == 'f':
            # compare floats for almost equality
            assert_array_almost_equal(o1, o2, decimal)
        else:
            # compare everything else (ints, bools) for absolute equality
            assert_array_equal(o1, o2)
    # probably these are other specialized types
    # that need a special check?
    else:
        eq_(o1, o2)


def assert_dict_equal(t1, t2, decimal=6):
    """
    Assert two dicts are equal.
    This method should actually
    work for any dict of numpy arrays/objects
    """

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
            eq_(val, t2[key])


def assert_sparse_matrix_equal(m1, m2, decimal=6):
    """Assert two scipy.sparse matrices are equal."""

    # delay the import to speed up stuff if this method is unused
    from scipy.sparse import isspmatrix
    from numpy.linalg import norm

    # both are sparse matricies
    assert isspmatrix(m1)
    assert isspmatrix(m2)

    # make sure they have the same format
    eq_(m1.format, m2.format)

    #make sure they have the same shape
    eq_(m1.shape, m2.shape)

    # even though its called assert_array_almost_equal, it will
    # work for scalars
    m1 = m1.tocsr()
    m2 = m2.tocsr()
    m1.eliminate_zeros()
    m2.eliminate_zeros()

    xi1, yi1 = m1.nonzero()
    xi2, yi2 = m2.nonzero()

    assert_array_equal(xi1, xi2)
    assert_array_equal(yi1, yi2)
    assert_array_almost_equal(m1.data, m2.data, decimal=decimal)

# decorator to mark tests as expected failure
def expected_failure(test):
    @functools.wraps(test)
    def inner(*args, **kwargs):
        try:
            test(*args, **kwargs)
        except BaseException:
            raise SkipTest
        else:
            raise AssertionError('Failure expected')
    return inner

# decorator to skip tests
def skip(rason):
    def wrap(test):
        @functools.wraps(test)
        def inner(*args, **kwargs):
            raise SkipTest
            print "After f(*args)"
        return inner
    return wrap


