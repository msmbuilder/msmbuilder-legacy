import numpy as np
import numpy.testing as npt
import scipy.sparse
from common import expected_failure, skip

from msmbuilder import MSMLib

def test_get_count_matrix_from_assignments_1():
    
    assignments = np.zeros((10,10))
    
    val = MSMLib.get_count_matrix_from_assignments(assignments).todense()
    correct = np.matrix([[90.0]])
    
    npt.assert_equal(val, correct)


def test_get_count_matrix_from_assignments_2():
    np.random.seed(42)
    
    assignments = np.random.randint(3, size=(10,10))
    
    val = MSMLib.get_count_matrix_from_assignments(assignments).todense()
    
    correct = np.matrix([[ 11.,   9.,  10.],
                         [  9.,  17.,   7.],
                         [ 10.,   7.,  10.]])
    npt.assert_equal(val, correct)


def test_get_count_matrix_from_assignments_3():
    np.random.seed(42)
    assignments = np.random.randint(3, size=(10,10))

    val = MSMLib.get_count_matrix_from_assignments(assignments, lag_time=2, sliding_window=False).todense()
    npt.assert_equal(val, np.matrix([[ 5.,   3.,   4.],
                         [  2.,  12.,   3.],
                         [ 4.,   3.,   4.]]))
                         
    val = MSMLib.get_count_matrix_from_assignments(assignments, lag_time=2, sliding_window=True).todense()
    npt.assert_equal(val, np.matrix([[8.,   9.,  11.],
                          [ 5.,  18.,   6.],
                          [ 11.,   5.,   7.]]))                 



def test_estimate_rate_matrix_1():
    np.random.seed(42)
    assignments = np.random.randint(2, size=(10,10))
    counts = MSMLib.get_count_matrix_from_assignments(assignments)
    K = MSMLib.estimate_rate_matrix(counts, assignments).todense()

    correct = np.matrix([[-40.40909091,   0.5       ],
            [  0.33928571, -50.55357143]])
    npt.assert_almost_equal(K, correct)


def test_estimate_rate_matrix_1():
    np.random.seed(42)
    counts_dense = np.random.randint(100, size=(4,4))
    counts_sparse = scipy.sparse.csr_matrix(counts_dense)
    
    t_mat_dense = MSMLib.estimate_transition_matrix(counts_dense)
    t_mat_sparse = MSMLib.estimate_transition_matrix(counts_sparse)
    
    correct = np.array([[ 0.22368421,  0.40350877,  0.06140351,  0.31140351],
     [ 0.24193548,  0.08064516,  0.33064516,  0.34677419],
     [ 0.22155689,  0.22155689,  0.26047904,  0.29640719],
     [ 0.23469388,  0.02040816,  0.21428571,  0.53061224]])
     
    npt.assert_almost_equal(t_mat_dense, correct)
    npt.assert_almost_equal(t_mat_dense, np.array(t_mat_sparse.todense()))


def test_apply_mapping_to_assignments_1():
    l = 100
    assignments = np.random.randint(l, size=(10,10))
    mapping = np.ones(l)
    
    MSMLib.apply_mapping_to_assignments(assignments, mapping)
    
    npt.assert_equal(assignments, np.ones((10,10)))


def test_apply_mapping_to_assignments_2():
    "preseve the -1s"
    
    l = 100
    assignments = np.random.randint(l, size=(10,10))
    assignments[0, 0] = -1
    mapping = np.ones(l)
    
    correct = np.ones((10,10))
    correct[0, 0] = -1
    
    MSMLib.apply_mapping_to_assignments(assignments, mapping)
    
    npt.assert_equal(assignments, correct)


def test_ergodic_trim():
    counts = scipy.sparse.csr_matrix(np.matrix('2 1 0; 1 2 0; 0 0 1'))
    trimmed, mapping = MSMLib.ergodic_trim(counts)
    
    npt.assert_equal(trimmed.todense(), np.matrix('2 1; 1 2'))
    npt.assert_equal(mapping, np.array([0, 1, -1]))


def test_trim_states():
    
    # run the (just tested) ergodic trim
    counts = scipy.sparse.csr_matrix(np.matrix('2 1 0; 1 2 0; 0 0 1'))
    trimmed, mapping = MSMLib.ergodic_trim(counts)
    
    # now try the segmented method
    states_to_trim = MSMLib.ergodic_trim_indices(counts)
    trimmed_counts = MSMLib.trim_states(states_to_trim, counts, assignments=None)
    
    npt.assert_equal(trimmed.todense(), trimmed_counts.todense())


class test_build_msm(object):
    #(assignments, lag_time, n_states=None, symmetrize='MLE', sliding_window=True, trimming=True, get_populations=False)
    def setup(self):
        self.assignments = np.array(np.matrix('0 1 0 0 0 1 0 0 0 1; 0 0 0 0 1 0 1 1 1 0'))
        self.lag_time = 1
    
    @skip
    def test_1(self):
        c, rc, t, p, m = MSMLib.build_msm(self.assignments, self.lag_time, symmetrize='MLE')
        npt.assert_array_equal(c.todense(), np.matrix('7 5; 4 2'))
        npt.assert_array_almost_equal(rc.todense(),
            np.matrix([[ 6.46159184,  4.61535527],
                       [ 4.61535527,  2.30769762]]))
        npt.assert_array_almost_equal(t.todense(), 
            np.matrix([[ 0.58333689,  0.41666311],
                [ 0.66666474,  0.33333526]]))
        npt.assert_array_almost_equal(p, [ 0.61538595,  0.38461405])
        npt.assert_array_equal(m, [0,1])
    
    @skip
    def test_2(self):
        c, rc, t, p, m = MSMLib.build_msm(self.assignments, self.lag_time, symmetrize=None)
        npt.assert_array_equal(c.todense(), np.matrix('7 5; 4 2'))
        npt.assert_array_almost_equal(rc.todense(),
            np.matrix([[ 7,  5],
                       [ 4,  2]]))
        npt.assert_array_almost_equal(t.todense(), 
            np.matrix([[ 0.58333333,  0.41666667],
                [ 0.66666667,  0.33333333]]))
        assert p is None
        npt.assert_array_equal(m, [0,1])


    @expected_failure
    def test_2_point_1(self):
        "This doesn't work"
        # same as test_2, just with get_populations=True
        c, rc, t, p, m = MSMLib.build_msm(self.assignments, self.lag_time, symmetrize=None,
            get_populations=True)
        npt.assert_array_equal(c.todense(), np.matrix('7 5; 4 2'))
        npt.assert_array_almost_equal(rc.todense(),
            np.matrix([[ 7,  5],
                       [ 4,  2]]))
        npt.assert_array_almost_equal(t.todense(), 
            np.matrix([[ 0.58333333,  0.41666667],
                [ 0.66666667,  0.33333333]]))
        npt.assert_array_almost_equal(p, [ 0.61538595,  0.38461405])
        npt.assert_array_equal(m, [0,1])
    
    @skip
    def test_3(self):
        c, rc, t, p, m = MSMLib.build_msm(self.assignments, self.lag_time, symmetrize='Transpose')
        npt.assert_array_equal(c.todense(), np.matrix('7 5; 4 2'))
        npt.assert_array_almost_equal(rc.todense(),
            np.matrix([[ 7,  4.5],
                       [ 4.5,  2]]))
        npt.assert_array_almost_equal(t.todense(), 
            np.matrix([[ 0.60869565,  0.39130435],
                [ 0.69230769,  0.30769231]]))
        npt.assert_array_almost_equal(p, [ 0.63888889,  0.36111111])
        npt.assert_array_equal(m, [0,1])
    
    @skip
    def test_4(self):
        c, rc, t, p, m = MSMLib.build_msm(self.assignments, lag_time=2, symmetrize=None, sliding_window=True)
        npt.assert_array_equal(c.todense(), np.matrix('7 4; 3 2'))
        npt.assert_array_almost_equal(rc.todense(), np.matrix('7 4; 3 2'))
        npt.assert_array_almost_equal(t.todense(), 
            np.matrix([[ 0.63636364,  0.36363636],
                [  0.6,  0.4]]))
        assert p is None
        npt.assert_array_equal(m, [0,1])

def test_estimate_transition_matrix_1():
    np.random.seed(42)
    count_matrix = np.array([[6, 3, 7], [4, 6, 9], [2, 6, 7]])
    t = MSMLib.estimate_transition_matrix(count_matrix)
    npt.assert_array_almost_equal(t, np.array([[ 0.375     ,  0.1875    ,  0.4375    ],
                                               [ 0.21052632,  0.31578947,  0.47368421],
                                               [ 0.13333333,  0.4       ,  0.46666667]]))
    
def test_invert_assignments_1():
    pass

def test_renumber_states_1():
    a = np.random.randint(3, size=(2,10))
    a[np.where(a==0)] = 1
    a[0,0] = -1
    
    # since its inplace
    new_a = a.copy()
    mapping = MSMLib.renumber_states(new_a)
    
    assert new_a[0,0] == -1
    npt.assert_array_equal(np.where(a==2), np.where(new_a==1))
    npt.assert_array_equal(mapping, [1,2])
    npt.assert_array_equal(mapping[new_a][np.where(a!= -1)], a[np.where(a!= -1)])

    
def test_log_likelihood_1():
    pass

def test_mle_reversible_count_matrix():
    pass