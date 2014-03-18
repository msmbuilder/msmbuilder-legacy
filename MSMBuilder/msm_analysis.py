# This file is part of MSMBuilder.
#
# Copyright 2011 Stanford University
#
# MSMBuilder is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

from __future__ import print_function, absolute_import, division
from mdtraj.utils.six.moves import xrange
import sys
import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg
import scipy
import numpy as np
import multiprocessing
import warnings
from mdtraj import io
from msmbuilder.utils import uneven_zip
import logging
logger = logging.getLogger(__name__)

# Set this value to true (msm_analysis.DisableErrorChecking=True) to ignore
# Eigenvector calculation errors.  Useful if you need to process disconnected data.
DisableErrorChecking = False

def get_reversible_eigenvectors(t_matrix, k, populations=None, right=False, **kwargs):
    """Find the k largest left eigenvalues and eigenvectors of a reversible
    row-stochastic matrix, sorted by eigenvalue magnitude

    Parameters
    ----------
    t_matrix : sparse or dense matrix
         The row-stochastic transition probability matrix
    k : int
        The number of eigenpairs to calculate. The `k` eigenpairs with the
        highest eigenvalues will be returned.
    populations : np.ndarray, optional
        The equilibrium, stationary distribution of t_matrix. If not supplied,
        it can be re-computed from `t_matrix`. But this substantially increases
        the runtime of the routine, so if the stationary distribution is known,
        it's more efficient to supply it.
    right : bool, optional
        The default behavior is to compute the *left* eigenvectors of the
        transition matrix. This option may be invoked to instead compute the
        *right* eigenvectors.

    Other Parameters
    ----------------
    Additional keyword arguments are passed directly to scipy.sparse.linalg.eigsh.
    Refer to the scipy documentation for further details.

    Notes
    -----
    A reversible transition matrix is one that satisifies the detailed balance
    condition

    .. math::

        \pi_i T_{i,j} = \pi_j T_{j,i}

    A reversible transition matrix satisifies a number of special
    conditions. In particular, it is simmilar to a symmetric matrix
    :math:`S_{i, j} = \sqrt{\frac{pi_i}{\pi_j}} T_{i, j} = S_{j, i}`.
    This property enables a much more robust solution to the eigenvector
    problem, because of the superior numerical stability of hermetian
    eigensolvers.

    See Also
    --------
    get_eigenvectors : computes the eigenpairs of a general row-stochastic matrix,
        without requiring that the matrix be reversible.
    scipy.sparse.linalg.eigsh
    """
    check_dimensions(t_matrix)
    n = t_matrix.shape[0]
    if k > n:
        logger.warning("You cannot calculate %d eigenvectors from a %d x %d matrix." % (k, n, n))
        k = n
        logger.warning("Instead, calculating %d eigenvectors." % k)
    elif k >= n - 1  and scipy.sparse.issparse(t_matrix):
        logger.warning("ARPACK cannot calculate %d eigenvectors from a %d x %d matrix." % (k, n, n))
        k = n - 2
        logger.warning("Instead, calculating %d eigenvectors." % k)

    if populations is None:
        populations = np.real(scipy.sparse.linalg.eigs(t_matrix.T, k=1, which='LR')[1][:,0])
        populations = populations / np.sum(populations)

    # Get the left eigenvectors using the sparse hermetian eigensolver
    # on the symmemtrized transition matrix
    root_pi = populations**(0.5)
    root_pi_diag = scipy.sparse.diags(root_pi, offsets=0).tocsr()
    root_pi_diag_inv = scipy.sparse.diags(1.0 / root_pi, offsets=0).tocsr()
    symtrans = root_pi_diag.dot(t_matrix).dot(root_pi_diag_inv)

    values, vectors = scipy.sparse.linalg.eigsh(symtrans.T, k=k, which='LA', **kwargs)

    # Reorder the eigenpairs by descending eigenvalue
    order = np.argsort(-np.real(values))
    values = values[order]
    vectors = vectors[:, order]

    # the eigenvectors of symtrans are a rotated version of the eigenvectors
    # of transmat that we want
    vectors = (root_pi * vectors.T).T

    # normalize the first eigenvector (populations)
    vectors[:, 0] /= np.sum(vectors[:, 0])

    return np.real(values), np.real(vectors)


def get_eigenvectors(t_matrix, n_eigs, epsilon=.001, dense_cutoff=50, right=False, tol=1E-30):
    """Get the left eigenvectors of a transition matrix, sorted by
    eigenvalue magnitude

    Parameters
    ----------
    t_matrix : sparse or dense matrix
        transition matrix. if `T` is sparse, the sparse eigensolver will be used
    n_eigs : int
        How many eigenvalues to calculate
    epsilon : float, optional
        Throw error if `T` is not a stochastic matrix, with tolerance given by `Epsilon`
    dense_cutoff : int, optional
        use dense eigensolver if dimensionality is below this
    right : bool, optional
        if true, compute the right eigenvectors instead of the left
    tol : float, optional
        Convergence criterion for sparse eigenvalue solver.


    Returns
    -------
    eigenvalues : ndarray
        1D array of eigenvalues
    eigenvectors : ndarray
        2D array of eigenvectors

    Notes
    -----
    Left eigenvectors satisfy the relation :math:`V \mathbf{T} = \lambda V`
    Vectors are returned in columns of matrix.
    Too large a value of tol may lead to unstable results.  See GitHub issue #174.
    """

    check_transition(t_matrix, epsilon)
    check_dimensions(t_matrix)
    n = t_matrix.shape[0]
    if n_eigs > n:
        logger.warning("You cannot calculate %d Eigenvectors from a %d x %d matrix." % (n_eigs, n, n))
        n_eigs = n
        logger.warning("Instead, calculating %d Eigenvectors." % n_eigs)
    if n < dense_cutoff and scipy.sparse.issparse(t_matrix):
        t_matrix = t_matrix.toarray()
    elif n_eigs >= n - 1  and scipy.sparse.issparse(t_matrix):
        logger.warning("ARPACK cannot calculate %d Eigenvectors from a %d x %d matrix." % (n_eigs, n, n))
        n_eigs = n - 2
        logger.warning("Instead, calculating %d Eigenvectors." % n_eigs)

    # if we want the left eigenvectors, take the transpose
    if not right:
        t_matrix = t_matrix.transpose()

    if scipy.sparse.issparse(t_matrix):
        values, vectors = scipy.sparse.linalg.eigs(t_matrix.tocsr(), n_eigs, which="LR", maxiter=100000,tol=tol)
    else:
        values, vectors = scipy.linalg.eig(t_matrix)

    order = np.argsort(-np.real(values))
    e_lambda = values[order]
    e_vectors = vectors[:, order]

    check_for_bad_eigenvalues(e_lambda, cutoff_value=1 - epsilon)  # this is bad IMO --TJL

    # normalize the first eigenvector (populations)
    e_vectors[:, 0] /= sum(e_vectors[:, 0])

    e_lambda = np.real(e_lambda[0: n_eigs])
    e_vectors = np.real(e_vectors[:, 0: n_eigs])

    return e_lambda, e_vectors


def get_implied_timescales(assignments_fn, lag_times, n_implied_times=100, sliding_window=True, trimming=True, symmetrize=None, n_procs=1):
    """Calculate implied timescales in parallel using multiprocessing library.  Does not work in interactive mode.

    Parameters
    ----------
    AssignmentsFn : str
        Path to Assignments.h5 file on disk
    LagTimes : list
        List of lag times to calculate the timescales at
    NumImpledTimes : int, optional
        Number of implied timescales to calculate at each lag time
    Slide : bool, optional
        Use sliding window
    Trim : bool, optional
        Use ergodic trimming
    Symmetrize : {'MLE', 'Transpose', None}
        Symmetrization method
    nProc : int
        number of processes to use in parallel (multiprocessing

    Returns
    -------
    formatedLags : ndarray
        RTM 6/27 I'm not quite sure what the semantics of the output is. It's not
        trivial and undocummented.

    See Also
    --------
    MSMLib.mle_reversible_count_matrix : (MLE symmetrization)
    MSMLib.build_msm
    get_eigenvectors

    """
    pool = multiprocessing.Pool(processes=n_procs)

    # subtle bug possibility; uneven_zip will let strings be iterable, whicj
    # we dont want
    inputs = uneven_zip([assignments_fn], lag_times, n_implied_times,
        sliding_window, trimming, [symmetrize])
    result = pool.map_async(get_implied_timescales_helper, inputs)
    lags = result.get(999999)

    # reformat
    formatted_lags = []
    for i, (lag_time_array, implied_timescale_array) in enumerate(lags):
        for j, lag_time in enumerate(lag_time_array):
            implied_timescale = implied_timescale_array[j]
            formatted_lags.append([lag_time, implied_timescale])

    formatted_lags = np.array(formatted_lags)

    pool.close()

    return formatted_lags


def get_implied_timescales_helper(args):
    """Helper function to compute implied timescales with multiprocessing

    Does not work in interactive mode

    Parameters
    ----------
    assignments_fn : str
        Path to Assignments.h5 file on disk
    n_states : int
        Number of states
    lag_time : list
        List of lag times to calculate the timescales at
    n_implied_times : int, optional
        Number of implied timescales to calculate at each lag time
    sliding_window : bool, optional
        Use sliding window
    trimming : bool, optional
        Use ergodic trimming
    symmetrize : {'MLE', 'Transpose', None}
        Symmetrization method

    Returns
    -------
    lagTimes : ndarray
        vector of lag times
    impTimes : ndarray
        vector of implied timescales

    See Also
    --------
    MSMLib.build_msm
    get_eigenvectors
    """
    assignments_fn, lag_time, n_implied_times, sliding_window, trimming, symmetrize = args
    logger.info("Calculating implied timescales at lagtime %d" % lag_time)

    try:
        assignments = io.loadh(assignments_fn, 'arr_0')
    except KeyError:
        assignments = io.loadh(assignments_fn, 'Data')

    try:
        from msmbuilder import MSMLib

        counts = MSMLib.get_count_matrix_from_assignments(assignments, lag_time=lag_time,
                                                          sliding_window=sliding_window)
        rev_counts, t_matrix, populations, mapping = MSMLib.build_msm(counts, symmetrize, trimming)

    except ValueError as e:
        logger.critical(e)
        sys.exit(1)

    n_eigenvectors = n_implied_times + 1
    if symmetrize in ['MLE', 'Transpose']:
        e_values = get_reversible_eigenvectors(t_matrix, n_eigenvectors, populations=populations)[0]
    else:
        e_values = get_eigenvectors(t_matrix, n_eigenvectors, epsilon=1)[0]

    # Correct for possible change in n_eigenvectors from trimming
    n_eigenvectors = len(e_values)
    n_implied_times = n_eigenvectors - 1

    # make sure to leave off equilibrium distribution
    lag_times = lag_time * np.ones((n_implied_times))
    imp_times = -lag_times / np.log(e_values[1: n_eigenvectors])

    return (lag_times, imp_times)


def check_for_bad_eigenvalues(eigenvalues, decimal=5, cutoff_value=0.999999):
    """Ensure that all eigenvalues are less than or equal to one

    Having multiple eigenvalues of lambda>=1 suggests either non-ergodicity or
    numerical error.


    Parameters
    ----------
    Eigenvalues : ndarray
        1D array of eigenvalues to check
    decimal : deprecated (marked 6/27)
        this doesn't do anything
    CutoffValue: float, optional
        Tolerance used

    Notes
    ------
    Checks that the first eigenvalue is within `CutoffValue` of 1, and that the second
    eigenvalue is not greater than `CutoffValue`

    """

    if abs(eigenvalues[0] - 1) > 1 - cutoff_value:
        warnings.warn(("WARNING: the largest eigenvalue is not 1, "
            "suggesting numerical error.  Try using 64 or 128 bit precision."))

        if eigenvalues[1] > cutoff_value:
            warnings.warn(("WARNING: the second largest eigenvalue (x) is close "
            " to 1, suggesting numerical error or nonergodicity.  Try using 64 "
            "or 128 bit precision.  Your data may also be disconnected, in "
            "which case you cannot simultaneously model both disconnected "
            "components.  Try collecting more data or trimming the "
            " disconnected pieces."))


def project_observable_onto_transition_matrix(observable, tprob, num_modes=25):
    """
    Projects an observable vector onto a probability transition matrix's
    eigenmodes.

    The function first decomposes the matrix `tprob` into `num_modes`
    different eigenvectors, sorted by eigenvalue. Then, it returns the
    amplitude of the projection of the observable onto each of those
    eigenmodes.

    This projection gives an estimate of how strong an
    experimental signal will be see at each timescale - though the actual
    experimental response will also be modulated by the populations of
    states at play.

    Parameters
    ----------
    observable : array_like, float
        a one-dimensional array of the values of a given observable for
        each state in the MSM
    tprob : matrix
        the transition probability matrix
    num_modes : int
        the number of eigenmodes to calculate (the top ones, sorted by mag.)

    Returns
    -------
    timescales : array_like, float
        the timescales of each eigenmode, in units of the lagtime of `tprob`
    amplitudes : array_like, float
        the amplitudes of the projection of `observable` onto each mode

    Notes
    -----
    The stationary mode is always discarded
    The eigenvalues/vectors are calculated from scratch, so this function
        may take a little while to run
    """

    if num_modes + 1 > tprob.shape[0]:
        logger.warning("cannot get %d eigenmodes from a rank %d matrix", num_modes + 1, tprob.shape[0])
        logger.warning("Getting as many modes as possible...")
        num_modes = tprob.shape[0] - 1

    eigenvalues, eigenvectors = get_eigenvectors(tprob, num_modes + 1, right=True)

    # discard the stationary eigenmode
    eigenvalues = np.real(eigenvalues[1:])
    eigenvectors = np.real(eigenvectors[:, 1:])

    timescales = - 1.0 / np.log(eigenvalues)

    amplitudes = np.zeros(num_modes)
    for mode in range(num_modes):
        amplitudes[mode] = np.dot(observable, eigenvectors[:, mode])

    return timescales, amplitudes


def sample(transition_matrix, state, steps, traj=None, force_dense=False):
    """Generate a random sequence of states by propogating a transition matrix.

    Parameters
    ----------
    transition_matrix : sparse or dense matrix
        A transition matrix
    State : {int, None, ndarray}
        Starting state for trajectory. If State is an integer, it will be used
        as the initial state. If State is None, an initial state will be
        randomly chosen from an uniform distribution. If State is an array, it
        represents a probability distribution from which the initial
        state will be drawn. If a trajectory is specified (see Traj keyword),
        this variable will be ignored, and the last state of that trajectory
        will be used.
    Steps : int
        How many steps to generate.
    Traj : list, optional
        An existing trajectory (python list) can be input; results will be
        appended to it
    ForceDense : bool, deprecated
        Force dense arithmatic.  Can speed up results for small models (OBSOLETE).

    Returns
    -------
    Traj : list
        Sequence of states as a python list
    """

    check_transition(transition_matrix)
    check_dimensions(transition_matrix)
    n_states = transition_matrix.shape[0]

    if scipy.sparse.isspmatrix(transition_matrix):
        transition_matrix = transition_matrix.tocsr()

    # reserve room for the new trajectory (will be appended to an existing trajectory at the end if necessary)
    newtraj = [-1] * steps

    # determine initial state
    if traj is None or len(traj) == 0:
        if state is None:
            state = np.random.randint(n_states)
        elif isinstance(state, np.ndarray):
            state = np.where(scipy.random.multinomial(1, state / sum(state)) == 1)[0][0]
        newtraj[0] = state
        start = 1
    else:
        state = traj[-1]
        start = 0
    if not state < n_states:
        raise ValueError("Intial state is %s, but should be between 0 and %s." % (state, n_states - 1))

    # sample the Markov chain
    if isinstance(transition_matrix, np.ndarray):
        for i in xrange(start, steps):
            p = transition_matrix[state, :]
            state = np.where(scipy.random.multinomial(1, p) == 1)[0][0]
            newtraj[i] = state

    elif isinstance(transition_matrix, scipy.sparse.csr_matrix):
        if force_dense:
            # Lutz: this is the old code path that converts the row of transition probabilities to a dense array at each step.
            # With the optimized handling of sparse matrices below, this can probably be deleted altogether.
            for i in xrange(start, steps):
                p = transition_matrix[state, :].toarray().flatten()
                state = np.where(scipy.random.multinomial(1, p) == 1)[0][0]
                newtraj[i] = state
        else:
            for i in xrange(start, steps):
                # Lutz: slicing sparse matrices is very slow (compared to slicing ndarrays)
                # To avoid slicing, use the underlying data structures of the CSR format directly
                T = transition_matrix
                vals = T.indices[T.indptr[state]:T.indptr[state + 1]]   # the column indices of the non-zero entries are the possible target states
                p = T.data[T.indptr[state]:T.indptr[state + 1]]         # the data values of the non-zero entries are the corresponding probabilities
                state = vals[np.where(scipy.random.multinomial(1, p) == 1)[0][0]]
                newtraj[i] = state
    else:
        raise RuntimeError("Unknown matrix type: %s" % type(T))

    # return the new trajectory, or the concatenation of the old trajectory with the new one
    if traj is None:
        return newtraj

    traj.extend(newtraj)
    return traj


def propagate_model(transition_matrix, n_steps, initial_populations, observable_vector=None):
    """Propogate the time evolution of a population vector.

    Parameters
    ----------
    T : ndarray or sparse matrix
        A transition matrix
    NumSteps : int
        How many timesteps to iterate
    initial_populations : ndarray
        The initial population vector
    observable_vector : ndarray
        Vector containing the state-wise averaged property of some observable.
        Can be used to propagate properties such as fraction folded, ensemble
        average RMSD, etc.  Default: None

    Returns
    -------
    X : ndarray
        Final population vector, after propagation
    obslist : list
        list of floats of length equal to the number of steps, giving the mean value
        of the observable (dot product of `ObservableVector` and populations) at
        each timestep

    See Also
    --------
    sample
    scipy.sparse.linalg.aslinearoperator

    """
    check_transition(transition_matrix)

    if observable_vector == None:
        check_dimensions(transition_matrix, initial_populations)
    else:
        check_dimensions(transition_matrix, initial_populations, observable_vector)

    X = initial_populations.copy()
    obslist = []
    if scipy.sparse.issparse(transition_matrix):
        TC = transition_matrix.tocsr()
    else:
        TC = transition_matrix

    Tl = scipy.sparse.linalg.aslinearoperator(TC)

    for i in xrange(n_steps):
        X = Tl.rmatvec(X)
        if observable_vector is not None:
            obslist.append(sum(observable_vector * X))

    return X, obslist


def calc_expectation_timeseries(tprob, observable, init_pop=None, timepoints=10 ** 6, n_modes=100, lagtime=15.0):
    """
    Calculates the expectation value over time <A(t)> for some `observable`
    in an MSM. Does this by eigenvalue decomposition, according to the eq

    math :: \langle A \rangle (t) = \sum_{i=0}^N \langle p(0), \psi^L_i
            \rangle e^{ - \lambda_i t } \langle \psi^R_i, A \rangle

    Parameters
    ----------
    tprob : matrix
        The transition probability matrix (of size N) for the MSM.

    observable : array_like, float
        A len N array of the values A of the observable for each state.

    init_pop : array_like, float
        A len N array of the initial populations of each state. If None
        is passed, then the function will start from even populations in
        each state.

    timepoints : int
        The number of timepoints to calculate - the final timeseries will
        be in length LagTime x `timepoints`

    n_modes : int
        The number of eigenmodes to include in the calculation. This
        number will depend on the timescales involved in the relatation
        of the observable.

    Returns
    -------
    timeseries : array_like, float
        A timeseries of the observable over time, in units of the lag time
        of the transition matrix.
    """

    # first, perform the eigendecomposition
    lambd, psi_L = get_eigenvectors(tprob, n_modes, right=False)
    psi_L = np.real(psi_L)
    lambd = np.real(lambd)

    pos_ind = np.where(lambd > 0)[0]
    lambd = lambd[pos_ind]
    psi_L = psi_L[:, pos_ind]
    n_modes = len(lambd)
    logger.info("Found %d non-negative eigenvalues" % n_modes)

    # normalize eigenvectors
    pi = psi_L[:, 0]
    pi /= pi.sum()

    np.savetxt('calculated_populations.dat', pi)
    psi_R = np.zeros(psi_L.shape)
    for i in range(n_modes):
        psi_L[:, i] /= np.sqrt(np.sum(np.square(psi_L[:, i]) / pi))
        psi_R[:, i] = psi_L[:, i] / pi

    if lagtime:
        logger.info("Shortest timescale process included: %s", -lagtime / np.log(np.min(lambd)))

    # figure out the initial populations
    if init_pop == None:
        init_pop = np.ones(tprob.shape[0])
        init_pop /= init_pop.sum()
    assert np.abs(init_pop.sum() - 1.0) < 0.0001

    # generate the timeseries
    timeseries = np.zeros(timepoints)
    for i in range(n_modes):
        front = np.dot(init_pop, psi_R[:, i])
        back = np.dot(observable, psi_L[:, i])
        mode_decay = front * np.power(lambd[i], np.arange(timepoints)) * back
        timeseries += np.real(mode_decay)

    logger.info('The equilibrium value is %f, while the last time point calculated is %f', np.dot(pi, observable), timeseries[-1])

    return timeseries


def msm_acf(tprob, observable, timepoints, num_modes=10):
    """
    Calculate an autocorrelation function from an MSM.

    Rapid calculation of the autocorrelation of an MSM is
    performed via an eigenmode decomposition.

    Parameters
    ----------
    tprob : matrix
        Transition probability matrix
    observable : ndarray, float
        Vector representing the observable value for each state
    timepoints : ndarray, int
        The timepoints at which to calculate the decay, in units of lag
        times.
    num_modes : int (num_modes)
        The number of eigenmodes to employ. More modes, more accurate,
        but slower.

    Returns
    -------
    acf : ndarray, float
        The autocorrelation function.

    Notes
    -----
    Use statsmodels.tsa.stattools.acf if you want to calculate an ACF from a
    raw observable such as an RMSD trace.

    See Docs/ACF/acf.pdf for a derivation of this calculation.
    """

    eigenvalues, eigenvectors = get_eigenvectors(tprob, num_modes + 1)
    num_modes = len(eigenvalues) - 1

    populations = eigenvectors[:, 0]
    D = np.diag(populations ** -1.)

    # discard the stationary eigenmode
    eigenvalues = np.real(eigenvalues[1:])
    eigenvectors = np.real(eigenvectors[:, 1:])
    right_eigenvectors = D.dot(eigenvectors)

    eigenvector_normalizer = np.diag(right_eigenvectors.T.dot(eigenvectors))
    eigenvectors /= eigenvector_normalizer

    S = eigenvectors.T.dot(observable)  # Project observable onto left eigenvectors

    acf = np.array([(eigenvalues ** t).dot(S**2) for t in timepoints])

    acf /= (eigenvalues ** 0.).dot(S**2)  # Divide by the ACF at time zero.

    return acf

# ======================================================== #
# SOME UTILITY FUNCTIONS FOR CHECKING TRANSITION MATRICES
# ======================================================== #


def flatten(*args):
    """Return a generator for a flattened form of all arguments"""

    for x in args:
        if hasattr(x, '__iter__'):
            for y in flatten(*x):
                yield y
        else:
            yield x


def is_transition_matrix(t_matrix, epsilon=.00001):
    """Check for row normalization of a matrix

    Parameters
    ----------
    t_matrix : densee or sparse matrix
    epsilon : float, optional
        threshold for how close the row sums need to be to 1

    Returns
    -------
    truth : bool
        True if the 2-norm of error in the row sums is less than Epsilon.
    """

    n = t_matrix.shape[0]
    row_sums = np.array(t_matrix.sum(1)).flatten()
    if scipy.linalg.norm(row_sums - np.ones(n)) < epsilon:
        return True
    return False


def are_all_dimensions_same(*args):
    """Are all the supplied arguments the same size

    Find the shape of every input.

    Returns
    -------
    truth : boolean
        True if every matrix and vector is the same size.
    """

    m = len(args)
    dim_list = []
    for i in range(m):
        dims = scipy.shape(args[i])
        dim_list.append(dims)

    return len(np.unique(flatten(dim_list))) == 1


def check_transition(t_matrix, epsilon=0.00001):
    """Ensure that matrix is a row normalized stochastic matrix

    Parameters
    ----------
    t_matrix : dense or sparse matrix
    epsilon : float, optional
        Threshold for how close the row sums need to be to one


    Other Parameters
    ----------------
    DisableErrorChecking : bool
        If this flag (module scope variable) is set tot True, this function just
        passes.

    Raises
    ------
    NormalizationError
        If T is not a row normalized stochastic matrix

    See Also
    --------
    check_dimensions : ensures dimensionality
    is_transition_matrix : does the actual checking
    """

    if not DisableErrorChecking and not is_transition_matrix(t_matrix, epsilon):
        logger.critical(t_matrix)
        logger.critical("Transition matrix is not a row normalized"
                        " stocastic matrix. This is often caused by "
                        "either numerical inaccuracies or by having "
                        "states with zero counts.")


def check_dimensions(*args):
    """Ensure that all the dimensions of the inputs are identical

    Raises
    ------
    DimensionError
        If some of the supplied arguments have different dimensions from one
        another
    """

    if are_all_dimensions_same(*args) == False:
        raise RuntimeError("All dimensions are not the same")
