import numpy as np
from msmbuilder.testing import eq
import scipy.sparse

from msmbuilder import MSMLib


def test_get_count_matrix_from_assignments_1():

    assignments = np.zeros((10, 10),'int')

    val = MSMLib.get_count_matrix_from_assignments(assignments).todense()
    correct = np.matrix([[90.0]])

    eq(val, correct)


def test_get_count_matrix_from_assignments_2():
    np.random.seed(42)

    assignments = np.random.randint(3, size=(10, 10))

    val = MSMLib.get_count_matrix_from_assignments(assignments).todense()

    correct = np.matrix([[11., 9., 10.],
                         [9., 17., 7.],
                         [10., 7., 10.]])
    eq(val, correct)


def test_get_count_matrix_from_assignments_3():
    np.random.seed(42)
    assignments = np.random.randint(3, size=(10, 10))

    val = MSMLib.get_count_matrix_from_assignments(
        assignments, lag_time=2, sliding_window=False).todense()
    eq(val, np.matrix([[5., 3., 4.],
                       [2., 12., 3.],
                       [4., 3., 4.]]))

    val = MSMLib.get_count_matrix_from_assignments(
        assignments, lag_time=2, sliding_window=True).todense()
    eq(val, np.matrix([[8., 9., 11.],
                       [5., 18., 6.],
                       [11., 5., 7.]]))


def test_estimate_rate_matrix_1():
    np.random.seed(42)
    assignments = np.random.randint(2, size=(10, 10))
    counts = MSMLib.get_count_matrix_from_assignments(assignments)
    K = MSMLib.estimate_rate_matrix(counts, assignments).todense()

    correct = np.matrix([[-40.40909091, 0.5],
                         [0.33928571, -50.55357143]])
    eq(K, correct)


def test_estimate_rate_matrix_2():
    np.random.seed(42)
    counts_dense = np.random.randint(100, size=(4, 4))
    counts_sparse = scipy.sparse.csr_matrix(counts_dense)

    t_mat_dense = MSMLib.estimate_transition_matrix(counts_dense)
    t_mat_sparse = MSMLib.estimate_transition_matrix(counts_sparse)

    correct = np.array([[0.22368421, 0.40350877, 0.06140351, 0.31140351],
                        [0.24193548, 0.08064516, 0.33064516, 0.34677419],
                        [0.22155689, 0.22155689, 0.26047904, 0.29640719],
                        [0.23469388, 0.02040816, 0.21428571, 0.53061224]])

    eq(t_mat_dense, correct)
    eq(t_mat_dense, np.array(t_mat_sparse.todense()))


def test_apply_mapping_to_assignments_1():
    l = 100
    assignments = np.random.randint(l, size=(10, 10))
    mapping = np.ones(l)

    MSMLib.apply_mapping_to_assignments(assignments, mapping)

    eq(assignments, np.ones((10, 10)))


def test_apply_mapping_to_assignments_2():
    "preseve the -1s"

    l = 100
    assignments = np.random.randint(l, size=(10, 10))
    assignments[0, 0] = -1
    mapping = np.ones(l)

    correct = np.ones((10, 10))
    correct[0, 0] = -1

    MSMLib.apply_mapping_to_assignments(assignments, mapping)

    eq(assignments, correct)


def test_ergodic_trim():
    counts = scipy.sparse.csr_matrix(np.matrix('2 1 0; 1 2 0; 0 0 1'))
    trimmed, mapping = MSMLib.ergodic_trim(counts)

    eq(trimmed.todense(), np.matrix('2 1; 1 2'))
    eq(mapping, np.array([0, 1, -1]))


def test_trim_states():

    # run the (just tested) ergodic trim
    counts = scipy.sparse.csr_matrix(np.matrix('2 1 0; 1 2 0; 0 0 1'))
    trimmed, mapping = MSMLib.ergodic_trim(counts)

    # now try the segmented method
    states_to_trim = MSMLib.ergodic_trim_indices(counts)
    trimmed_counts = MSMLib.trim_states(
        states_to_trim, counts, assignments=None)

    eq(trimmed.todense(), trimmed_counts.todense())

    assignments = np.array([np.arange(counts.shape[0])])
    states_to_trim = MSMLib.ergodic_trim_indices(counts)
    trimmed_counts, trimmed_assignments = MSMLib.trim_states(states_to_trim, counts, assignments=assignments)  # Test that code works with assignments given
    
    trimmed_assignments_ref = np.array([[0, 1, -1]])  # State 2 is strong-disconnected so set to -1
    
    eq(trimmed_assignments, trimmed_assignments_ref)
    


class test_build_msm(object):

    def setup(self):
        self.assignments = np.array(
            np.matrix('0 1 0 0 0 1 0 0 0 1; 0 0 0 0 1 0 1 1 1 0'))
        self.lag_time = 1

    def test_1(self):

        C = MSMLib.get_count_matrix_from_assignments(self.assignments, 2)
        rc, t, p, m = MSMLib.build_msm(
            C, symmetrize='MLE', ergodic_trimming=True)

        eq(rc.todense(),
            np.matrix([[6.46159184, 4.61535527],
                       [4.61535527, 2.30769762]]), decimal=4)
        eq(t.todense(),
            np.matrix([[0.58333689, 0.41666311],
                       [0.66666474, 0.33333526]]), decimal=4)
        eq(p, np.array([0.61538595, 0.38461405]), decimal=5)
        eq(m, np.array([0, 1]))


def test_estimate_transition_matrix_1():
    np.random.seed(42)
    count_matrix = np.array([[6, 3, 7], [4, 6, 9], [2, 6, 7]])
    t = MSMLib.estimate_transition_matrix(count_matrix)
    eq(t, np.array([[0.375, 0.1875, 0.4375],
                    [0.21052632, 0.31578947, 0.47368421],
                    [0.13333333, 0.4, 0.46666667]]))


def test_invert_assignments_1():
    pass


def test_renumber_states_1():
    a = np.random.randint(3, size=(2, 10))
    a[np.where(a == 0)] = 1
    a[0, 0] = -1

    # since its inplace
    new_a = a.copy()
    mapping = MSMLib.renumber_states(new_a)

    eq(int(new_a[0, 0]), -1)
    eq(np.where(a == 2)[0], np.where(new_a == 1)[0])
    eq(np.where(a == 2)[1], np.where(new_a == 1)[1])
    eq(mapping, np.array([1, 2]))
    eq(mapping[new_a][np.where(a != -1)], a[np.where(a != -1)])


def test_log_likelihood_1():
    pass


class test_mle_reversible_count_matrix(object):
    def test_1(self):
        """Checks that the MLE code (kyle) agrees with the reference
        implemenation (Lutz).
        """
        counts = scipy.sparse.csr_matrix(
            [[10, 5, 1], [0, 25, 10], [3, 3, 100]])
        self.compare_kyle_to_lutz(counts)

    def test_2(self):
        """Checks that a symmetric matrix is unchanged by the MLE"""
        counts = scipy.sparse.csr_matrix([[10, 5], [5, 10]])
        self.compare_kyle_to_reference(counts, counts)

    def compare_kyle_to_lutz(self, raw_counts):
        """Kyle wrote the most recent MLE code.  We compare to the previous
        code that was written by Lutz.
        """

        counts = MSMLib.ergodic_trim(raw_counts)[0]

        x_kyle = MSMLib.mle_reversible_count_matrix(counts)
        x_kyle /= x_kyle.sum()

        x_lutz = MSMLib.__mle_reversible_count_matrix_lutz__(counts)
        x_lutz /= x_lutz.sum()

        eq(x_kyle.toarray(), x_lutz.toarray())

    def compare_kyle_to_reference(self, raw_counts, reference):
        """Compare MLE estimated reversible counts to a reference matrix"""
        counts = MSMLib.ergodic_trim(raw_counts)[0]
        x_kyle = MSMLib.mle_reversible_count_matrix(counts)

        eq(x_kyle.toarray(), reference.toarray())
