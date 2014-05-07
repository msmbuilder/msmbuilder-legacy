from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from msmbuilder.MSMLib import build_msm
from msmbuilder.msm_analysis import get_eigenvectors, get_reversible_eigenvectors

def test_get_eigenvectors_left():    
    # just some random counts
    N = 100
    counts = np.random.randint(1, 10, size=(N,N))
    transmat, pi = build_msm(scipy.sparse.csr_matrix(counts), 'MLE')[1:3]

    values0, vectors0 = get_eigenvectors(transmat, 10)
    values1, vectors1 = get_reversible_eigenvectors(transmat, 10)
    values2, vectors2 = get_reversible_eigenvectors(transmat, 10, populations=pi)

    # check that the eigenvalues are the same using the two methods
    np.testing.assert_array_almost_equal(values0, values1)
    np.testing.assert_array_almost_equal(vectors0, vectors1)
    np.testing.assert_array_almost_equal(vectors1, vectors2)
    
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

def test_get_eigenvectors_right():
    # just some random counts
    N = 100
    counts = np.random.randint(1, 10, size=(N,N))
    transmat, pi = build_msm(scipy.sparse.csr_matrix(counts), 'MLE')[1:3]

    values0, vectors0 = get_eigenvectors(transmat, 10, right=True)
    values1, vectors1 = get_reversible_eigenvectors(transmat, 10, right=True)
    values2, vectors2 = get_reversible_eigenvectors(transmat, 10, right=True, populations=pi)

    # check that the eigenvalues are the same using the two methods
    np.testing.assert_array_almost_equal(values0, values1)
    np.testing.assert_array_almost_equal(vectors0, vectors1)
    np.testing.assert_array_almost_equal(vectors1, vectors2)

    # check that the eigenvectors returned by both methods are _actually_
    # left eigenvectors of the transmat
    def test_eigenpairs(values, vectors):
        for value, vector in zip(values, vectors.T):
            np.testing.assert_array_almost_equal(
                (transmat.dot(vector) / vector).flatten(), np.ones(N)*value)

    ones_ary = np.ones(pi.shape)
    np.testing.assert_array_almost_equal(ones_ary, vectors0[:, 0])
    np.testing.assert_array_almost_equal(ones_ary, vectors1[:, 0])
    np.testing.assert_array_almost_equal(ones_ary, vectors2[:, 0])
    test_eigenpairs(values0, vectors0)
    test_eigenpairs(values1, vectors1)
    test_eigenpairs(values2, vectors2)


def test_eigenvector_norm():
    N = 100
    counts = np.random.randint(1, 10, size=(N,N))
    transmat, pi = build_msm(scipy.sparse.csr_matrix(counts), 'MLE')[1:3]

    left_values0, left_vectors0 = get_eigenvectors(transmat, 10, right=False, normalized=True)
    right_values0, right_vectors0 = get_eigenvectors(transmat, 10, right=True, normalized=True)

    left_values1, left_vectors1 = get_reversible_eigenvectors(transmat, 10, right=False, normalized=True)
    right_values1, right_vectors1 = get_reversible_eigenvectors(transmat, 10, right=True, normalized=True)

    np.testing.assert_array_almost_equal(left_values0, right_values0)
    np.testing.assert_array_almost_equal(left_values1, right_values1)

    Id = np.eye(10)

    np.testing.assert_array_almost_equal(left_vectors0.T.dot(right_vectors0), Id)
    np.testing.assert_array_almost_equal(left_vectors1.T.dot(right_vectors1), Id)
    # ^^^ this tests whether they're normalized and also orthogonal
    # it's probably redundant to do both since the previous two tests look
    # to see if the left_vectors and right_vectors are the same
    # doesn't hurt to check in case normalizing screws things up
