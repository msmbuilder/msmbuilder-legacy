from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from msmbuilder.MSMLib import build_msm
from msmbuilder.msm_analysis import get_eigenvectors, get_reversible_eigenvectors

def test_get_eigenvectors():    
    # just some random counts
    N = 100
    counts = np.random.randint(1, 10, size=(N,N))
    transmat, pi = build_msm(scipy.sparse.csr_matrix(counts), 'MLE')[1:3]

    values0, vectors0 = get_eigenvectors(transmat, 10)
    values1, vectors1 = get_reversible_eigenvectors(transmat, 10)
    values2, vectors2 = get_reversible_eigenvectors(transmat, 10, populations=pi)

    # check that the eigenvalues are the same using the two methods
    np.testing.assert_array_almost_equal(values0, values1)
    
    # check that the eigenvectors returned by both methods are _actually_
    # left eigenvectors of the transmat
    def test_eigenpairs(values, vectors):
        for value, vector in zip(values, vectors.T):
            np.testing.assert_array_almost_equal(
                (transmat.T.dot(vector) / vector).flatten(), np.ones(N)*value)

    np.testing.assert_array_almost_equal(pi, vectors0[:, 0])
    np.testing.assert_array_almost_equal(pi, vectors1[:, 0])
    np.testing.assert_array_almost_equal(pi, vectors2[:, 0])
    test_eigenpairs(values0, vectors0)
    test_eigenpairs(values1, vectors1)
    test_eigenpairs(values2, vectors2)
