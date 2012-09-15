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

"""Classes and functions for working with Transition and Count Matrices.

Notes

* Assignments typically refer to a numpy array of integers such that Assignments[i,j] gives the state of trajectory i, frame j.
* Transition and Count matrices are typically stored in scipy.sparse.csr_matrix format.

MSMLib functions generally relate to one of the following

* Counting the number of transitions observed in Assignment data--e.g., constructing a Count matrix.
* Constructing a transition matrix from a count matrix.
* Performing calculations with Assignments, Counts matrices, or transition matrices.

"""
import warnings
import scipy.sparse
import scipy.linalg
import scipy
import numpy as np
import scipy.optimize
from collections import defaultdict
from msmbuilder.utils import deprecated
from msmbuilder import msm_analysis
import logging
logger = logging.getLogger('MSMLib')

def estimate_rate_matrix(count_matrix, assignments):
    """MLE Rate Matrix given transition counts and *dwell times*
    
    Parameters
    ----------
    count_matrix : sparse or dense matrix
        transition counts
    assignments : ndarray
        2D assignments array used to compute average dwell times
    
    Returns
    -------
    K : csr_matrix
        Rate matrix
    
    Notes
    -----
    The *correct* likelihood function to use for estimating the rate matrix when
    the data is sampled at a discrete frequency is open for debate. This
    likelihood function doesn't take into account the error in the lifetime estimates
    from the discrete sampling. Other methods are currently under development (RTM 6/27)
    
    See Also
    --------
    EstimateTransitionMatrix
    
    References
    ----------
    .. [1] Buchete NV, Hummer G. "Coarse master equaions for peptide folding
        dynamics." J Phys Chem B 112:6057-6069.
    """
    
    # Find the estimated dwell times (need to deal w negative ones)
    neg_ind = np.where(assignments == -1)
    n = np.max(assignments.flatten()) + 1
    assignments[neg_ind] = n
    R = np.bincount(assignments.flatten())
    R = R[:n]
    assert count_matrix.shape[0] == R.shape[0]
    
    # Most Likely Estimator ( Kij(hat) = Nij / Ri )
    if scipy.sparse.isspmatrix(count_matrix):
        C = scipy.sparse.csr_matrix(count_matrix).asfptype()
        D = scipy.sparse.dia_matrix((1.0 / R, 0), C.shape).tocsr()
        K = D*C # if all is sparse is matrix multiply, formerly: D.dot( C )
    else:
        # deprecated due to laziness --TJL
        raise ValueError("ERROR! Pass sparse matrix to me")
   
    # Now get the diagonals right. They should be negative row sums
    row_sums = np.asarray(C.sum(axis=1)).flatten()
    current  = K.diagonal()
    S = scipy.sparse.dia_matrix(((row_sums + (2.0*current)), 0), C.shape).tocsr()
    K = K - S
    if not K.shape == count_matrix.shape:
        raise RuntimeError('Bad news bears')
    #assert K.sum(0).all() == np.zeros(K.shape[0]).all(), K.sum(0).all()
 
    return K


def estimate_transition_matrix(count_matrix):
    """
    Simple Maximum Likelihood estimator of transition matrix.
    
    Parameters
    ----------
    count_matrix : array or sparse matrix
        A square matrix of transition counts
    
    Returns
    -------
    tProb : array or sparse matrix
         Most likely transition matrix given `tCount`
    """
    #1.  Make sure you don't modify tCounts.
    #2.  Make sure you handle both floats and ints 
    if scipy.sparse.isspmatrix(count_matrix):
        C = scipy.sparse.csr_matrix(count_matrix).asfptype()
        weights = np.asarray(C.sum(axis=1)).flatten()            
        inv_weights = np.zeros( len(weights) )
        inv_weights[ weights != 0 ] = 1.0 / weights[ weights != 0 ]
        D = scipy.sparse.dia_matrix((inv_weights,0),C.shape).tocsr()
        tProb = D.dot(C)
    else:
        tProb = np.asarray(count_matrix.astype(float)) # astype creates a copy
        weights = tProb.sum(axis=1)
        inv_weights = np.zeros( len(weights) )
        inv_weights[ weights != 0 ] = 1.0 / weights[ weights != 0 ]
        tProb = tProb * inv_weights.reshape((weights.shape[0],1))
    
    return tProb


def build_msm(assignments, lag_time, symmetrize='MLE', sliding_window=True, trim=True):
    """
    Given trajectories in a discrete state space (`assignments`), estimate the
    transition probability matrix governing dynamics on that space.
    
    Specifically, this function does a number of things (in order):
        (1) Count the transitions in `assignments` at a specific lag time
        (2) Optionally employ Tarjan's algorithm to discard unconnected
            components of the network.
        (3) Optionally modify the estimate of the transitions to account for
            the fact that the system dynamics are reversible.
        (4) Estimate the most likely transition probability matrix given the
            data.
        (5) Compute the equilibirum populations of each state dicated by the
            transition matrix.
    
    Parameters
    ----------
    assignments : np.ndarray, int
        2D msmbuilder assignments array
    lag_time :
        the lag time to build the msm with, in units of frames
    symmetrize : {'MLE', 'Transpose', None}, optional
        symmetrization scheme to apply to estimate reversiblity
    sliding_window : bool, optional
        Whether or not to employ sliding window esimation in the counting 
        of transitions. If True, all frames of reference are used when 
        counting, if false, only the first frame of reference is used.
    trim : bool, optional
        If true, trims out unconnected components of the MSM network
    
    Returns
    -------
    counts : scipy.sparse.csr_matric
        The counts matrix for the MSM, with reversibility applied.
    t_matrix : scipy.sparse_csr_matrix
        The most likely transition matrix, given counts
    populations : np.ndarray, float
        The equilibirum populations of each state
    mapping : np.ndarray
        A vector mapping the old state indices in `assignments` to
        the new state indices in the objects returned. This is
        necessary since the `trim`ing procedue can remove states.
    """
    
    counts = get_count_matrix_from_assignments(assignments, lag_time=lag_time, 
                                               sliding_window=sliding_window)
    if trim:
        counts, mapping = ergodic_trim(counts)
    else:
        mapping = np.arange( counts.shape[0] )
    
    # Apply a symmetrization scheme
    t_matrix, counts = build_msm_from_counts(counts, lag_time, 
                                             symmetrize, return_rev_counts=True)
    
    # compute the equilibrium populations
    if symmetrize in ['mle', 'transpose']:
        populations = np.array(rev_counts.sum(0)).flatten()
    else:
        vectors = msm_analysis.get_eigenvectors(t_matrix, 5)[1]
        populations = vectors[:, 0]
    populations /= populations.sum()
    
    return counts, t_matrix, populations, mapping


def build_msm_from_counts(counts, lag_time, symmetrize, return_rev_counts=False):
    """
    Estimates the transition probability matrix from the counts matrix.
    
    Parameters
    ----------
    counts : matrix
        the MSM counts matrix
    lag_time :
        the lag time to build the msm with, in frames
    symmetrize : {'MLE', 'Transpose', None}
        symmetrization scheme so that we have reversible counts
    return_rev_counts : bool
        whether or not to return the reversible counts or not
        
        
    Returns
    -------
    t_matrix : matrix
        the transition probability matrix
    rev_counts : matrix
        the estimate of the reversible counts 
        (only returned if `return_rev_counts` is True)
    """
    
    symmetrize = str(symmetrize).lower()
    symmetrization_error = ValueError("Invalid symmetrization scheme requested: %s. Exiting." % symmetrize)
    if symmetrize not in ['mle', 'transpose', 'none']:
        raise symmetrization_error
    
    if symmetrize == 'mle':
        rev_counts = mle_reversible_count_matrix(counts, prior=0.0)

    elif symmetrize == 'transpose':
        rev_counts = 0.5*(counts + counts.transpose())

    elif symmetrize == 'none':
        rev_counts = counts

    else:
        raise symmetrization_error
    
    t_matrix = estimate_transition_matrix(rev_counts)
    
    if return_rev_counts:
        return t_matrix, rev_counts
    else:
        return t_matrix


def get_count_matrix_from_assignments(assignments, n_states=None, lag_time=1, sliding_window=True):
    """Calculate count matrix from Assignments.
    
    Parameters
    ----------
    assignments : ndarray
        A 2d ndarray containing the state assignments.  
    n_states : int, optional
        Can be automatically determined, unless you want a model with more states than are observed
    lag_time: int, optional
        the LagTime with which to estimate the count matrix. Default: 1
    sliding_window: bool, optional
        Use a sliding window.  Default: True
        
    Returns
    -------
    counts : sparse matrix
        `Counts[i,j]` stores the number of times in the assignments that a
        trajectory went from state i to state j in `LagTime` frames
        
    Notes
    -----
    assignments are input as iterables over numpy 1-d arrays of integers.
    For example a 2-d array where assignments[i,j] gives the ith trajectory, jth frame.
    The beginning and end of each trajectory may be padded with negative ones, which will be ignored.
    If the number of states is not given explitly, it will be determined as one plus the largest state index of the Assignments.
    Sliding window yields non-independent samples, but wastes less data.
    """
    
    if not n_states:
        n_states = 1 + int(np.max([np.max(a) for a in assignments]))   # Lutz: a single np.max is not enough, b/c it can't handle a list of 1-d arrays of different lengths
        if n_states < 1:
            raise ValueError()
        
    C = scipy.sparse.lil_matrix((int(n_states), int(n_states)), dtype='float32')  # Lutz: why are we using float for count matrices?
    
    for A in assignments:
        FirstEntry = np.where(A!=-1)[0]
        if len(FirstEntry) >= 1: #New Code by KAB to skip pre-padded negative ones.  This should solve issues with Tarjan trimming results.
            FirstEntry = FirstEntry[0]
            A = A[FirstEntry:]
            C = C + get_counts_from_traj(A, n_states, lag_time=lag_time, sliding_window=sliding_window)#.tolil()
    
    return C


def get_counts_from_traj(states, n_states=None, lag_time=1, sliding_window=True):
    """Computes the transition count matrix for a sequence of states (single trajectory).
    
    Parameters
    ----------
    states : array
        A one-dimensional array of integers representing the sequence of states.
        These integers must be in the range [0, n_states]
    n_states : int
        The total number of states. If not specified, the largest integer in the
        states array plus one will be used.
    lag_time : int, optional
        The time delay over which transitions are counted
    sliding_window : bool, optional
        Use sliding window
        
    Returns
    -------
    C : sparse matrix of integers
        The computed transition count matrix
    """
    
    if not n_states:
        n_states = np.max(states)+1
        
    if sliding_window:
        from_states = states[:-lag_time:1]
        to_states = states[lag_time::1]
    else:
        from_states = states[:-lag_time:lag_time]
        to_states = states[lag_time::lag_time]
    assert from_states.shape == to_states.shape
    
    transitions = np.row_stack((from_states, to_states))
    counts = np.ones(transitions.shape[1], dtype=int)
    try:
        C = scipy.sparse.coo_matrix((counts, transitions),
            shape=(n_states, n_states))
    except ValueError:
        # Lutz: if we arrive here, there was probably a state with index -1
        # we try to fix it by ignoring transitions in and out of those states
        # (we set both the count and the indices for those transitions to 0)
        mask = transitions < 0
        counts[mask[0, :] | mask[1, :]] = 0
        transitions[mask] = 0
        C = scipy.sparse.coo_matrix((counts, transitions),
            shape=(n_states, n_states))
            
    return C


def apply_mapping_to_assignments(assignments, mapping):
    """Remap the states in an assignments file according to a mapping.
    
    Parameters
    ----------
    assignments : ndarray
        Standard 2D assignments array
    mapping : ndarray
        1D numpy array of length equal to the number of states in Assignments.
        Mapping[a] = b means that the frames currently in state a will be mapped
        to state b
    
    Returns
    -------
    NewAssignments : ndarray 
    
    Notes
    -----
    This function is useful after performing PCCA or Ergodic Trimming. Also, the
    state -1 is treated specially -- it always stays -1 and is not remapped.
    
    """
    
    NewMapping = mapping.copy()
    # Make a special state for things that get deleted by Ergodic Trimming.
    NewMapping[np.where(mapping == -1)] = mapping.max() + 1
    
    NegativeOneStates = np.where(assignments == -1)
    assignments[:] = NewMapping[assignments]
    WhereEliminatedStates = np.where(assignments == (mapping.max() + 1))
    
    # These are the dangling 'tails' of trajectories (with no actual data) that we denote state -1.
    assignments[NegativeOneStates] = -1
    # These states have typically been "deleted" by the ergodic trimming algorithm.  Can be at beginning or end of trajectory.
    assignments[WhereEliminatedStates] = -1


def invert_assignments(assignments):
    """Invert an assignments array -- that is, produce a mapping
    from state -> traj/frame
    
    Parameters
    ----------
    assignments : np.ndarray
        2D array of MSMBuilder assignments
    
    Returns
    -------
    inverse_mapping : collections.defaultdict
        Mapping from state -> traj,frame, such that inverse_mapping[s]
        gives the conformations assigned to state s.
    
    """
    
    inverse_mapping = defaultdict(lambda: ([], []))
    for i in xrange(assignments.shape[0]):
        for j in xrange(assignments.shape[1]):
            inverse_mapping[assignments[i, j]][0].append(i)
            inverse_mapping[assignments[i, j]][1].append(j)
    
    return inverse_mapping


def apply_mapping_to_vector(vector, mapping):
    """ Remap an observable vector after ergodic trimming
    
    RTM 6/27: I don't think this function is really doing what it should.
    It does a reordering, but when the mapping is a many->one, don't you really
    want to average things together or something?
    
    TJL 7/1: That's true. I wrote this with only the ergodic trimming in
    mind, it needs to be updated if it's going to work w/PCCA as well...
    
    Parameters
    ----------
    vector : ndarray
        1D. Some observable value associated with each states
    Mapping : ndarray
        1D numpy array of length equal to the number of states in Assignments.
        Mapping[a] = b means that the frames currently in state a are now assigned
        to state b, and thus their observable should be too
    
    Returns
    -------
    new_vector : ndarray
        mapped observable values
    
    Notes
    -----
    The state -1 is treated specially -- it always stays -1 and is not remapped.
    
    """
    
    new_vector = vector[np.where(mapping != -1)[0]] 
    logger.info("Mapping %d elements --> %d", len(vector), len(new_vector))
    
    return new_vector


def renumber_states(assignments):
    """Renumber states to be consecutive integers (0, 1, ... , n), performs
    this transformation in place.
    
    Parameters
    ----------
    assignments : ndarray
        2D array of msmbuilder assignments
    
    Returns
    -------
    mapping : ndarray, int
        A mapping from the old numbering scheme to the new, such that
        mapping[new] = old
    
    Notes
    -----
    Useful if some states have 0 counts.
    """
    
    unique = list(np.unique(assignments))
    if unique[0] == -1:
        minus_one = np.where(assignments == -1)
        unique.pop(0)
    else:
        minus_one = []
    
    inverse_mapping = invert_assignments(assignments)
    
    for i, x in enumerate(unique):
        assignments[inverse_mapping[x]] = i
    assignments[minus_one] = -1
    
    mapping = np.array(unique, dtype=int)
    return mapping


def tarjan(graph):
    """Find the strongly connected components in a graph using Tarjan's algorithm.
    
    Parameters
    ----------
    graph : dict
        mapping from node names to lists of successor nodes.
    
    Returns
    -------
    components : list
        list of the strongly connected components
    
    Notes
    -----
    Code based on ActiveState code by Josiah Carlson (New BSD license).
    Most users will want to call the ErgodicTrim() function rather than directly calling Tarjan().
    
    See Also
    --------
    ErgodicTrim
    
    """
    n_states = graph.shape[0]
    
    #Keeping track of recursion state info by node
    Nodes = np.arange(n_states)
    NodeNums = [None for i in range(n_states)]
    NodeRoots = np.arange(n_states)
    NodeVisited = [False for i in range(n_states)]
    NodeHidden = [False for i in range(n_states)]
    NodeInComponent = [None for i in range(n_states)]
    
    stack = []
    components = []
    nodes_visit_order = []
    graph.next_visit_num = 0
    
    def visit(v):
        "Mark a state as visited"
        call_stack = [(1, v, graph.getrow(v).nonzero()[1], None)]
        while call_stack:
            tovisit, v, iterator, w = call_stack.pop()
            if tovisit:
                NodeVisited[v] = True
                nodes_visit_order.append(v)
                NodeNums[v] = graph.next_visit_num
                graph.next_visit_num += 1
                stack.append(v)
            if w and not NodeInComponent[v]:
                NodeRoots[v] = nodes_visit_order[min(NodeNums[NodeRoots[v]],
                                                     NodeNums[NodeRoots[w]])]
            cont = 0
            for w in iterator:
                if not NodeVisited[w]:
                    cont = 1
                    call_stack.append((0, v, iterator, w))
                    call_stack.append((1, w, graph.getrow(w).nonzero()[1], None))
                    break
                if not NodeInComponent[w]:
                    NodeRoots[v] = nodes_visit_order[min(NodeNums[NodeRoots[v]],
                                                         NodeNums[NodeRoots[w]])]
            if cont:
                continue
            if NodeRoots[v] == v:
                c = []
                while 1:
                    w = stack.pop()
                    NodeInComponent[w] = c
                    c.append(w)
                    if w == v:
                        break
                components.append(c)
    # the "main" routine
    for v in Nodes:
        if not NodeVisited[v]:
            visit(v)
    
    # extract SCC info
    for n in Nodes:
        if NodeInComponent[n] and len(NodeInComponent[n]) > 1:
            # part of SCC
            NodeHidden[n] = False
        else:
            # either not in a component, or singleton case
            NodeHidden[n] = True
    
    return(components)


def ergodic_trim(counts, assignments=None):
    """Use Tarjan's Algorithm to find maximal strongly connected subgraph.
    
    Parameters
    ----------
    counts : csr sparse matrix
        transition counts
    assignments : ndarray, optional
        Optionally map assignments to the new states, nulling out disconnected regions.
        
    Notes
    -----
    The component with maximum number of counts is selected
    
    See Also
    --------
    Tarjan
    
    """
    
    NZ = np.array(counts.nonzero()).transpose()
    
    ConnectedComponents = tarjan(counts)
    PiSym = np.array(counts.sum(0)).flatten()
    ComponentPops = np.array([sum(PiSym[np.array(x)]) for x in ConnectedComponents])
    ComponentInd = np.argmax(ComponentPops)
    
    logger.info("Selected component %d with population %f", ComponentInd, ComponentPops[ComponentInd] / ComponentPops.sum())
    
    GoodComponent = np.unique(ConnectedComponents[ComponentInd])
    
    Mapping = np.zeros(counts.shape[0], dtype='int') - 1
    for i, x in enumerate(GoodComponent):
        Mapping[x] = i
        
    NZ[:, 0] = Mapping[NZ[:, 0]]
    NZ[:, 1] = Mapping[NZ[:, 1]]
    
    Ind = np.where(NZ.min(1) != -1)
    X = scipy.sparse.csr_matrix((counts.data[Ind], NZ[Ind].transpose()))
    
    if assignments is not None:
        apply_mapping_to_assignments(assignments, Mapping)
    
    return (X, Mapping)


def log_likelihood(count_matrix, transition_matrix):
    """log of the likelihood of an observed count matrix given a transition matrix
    
    Parameters
    ----------
    count_matrix : ndarray or sparse matrix
        Transition count matrix.
    transition_matrix : ndarray or sparse matrix
        Transition probability matrix.
        
    Returns
    -------
    loglikelihood : float
        The natural log of the likelihood, computed as
        :math:`\sum_{ij} C_{ij} \log(P_{ij})`
            
    
    """
    
    if isinstance(transition_matrix, np.ndarray) and isinstance(count_matrix, np.ndarray):
        # make sure that both count_matrix and transition_matrix are arrays
        count_matrix = np.asarray(count_matrix)
        # (not dense matrices), so we can use element-wise multiplication
        transition_matrix = np.asarray(transition_matrix)
        mask = count_matrix > 0
        return np.sum(np.log(transition_matrix[mask]) * count_matrix[mask])
        
    else:
        # make sure both count_matrix and transition_matrix are sparse CSR matrices
        if not scipy.sparse.isspmatrix(count_matrix):
            count_matrix = scipy.sparse.csr_matrix(count_matrix)
        else:
            count_matrix = count_matrix.tocsr()
            
        if not scipy.sparse.isspmatrix(transition_matrix):
            transition_matrix = scipy.sparse.csr_matrix(transition_matrix)
        else:
            transition_matrix = transition_matrix.tocsr()
        row, col = count_matrix.nonzero()
        
        return np.sum(np.log(np.asarray(transition_matrix[row, col])) \
            * np.asarray(count_matrix[row, col]))


def mle_reversible_count_matrix(count_matrix, prior=0.0, initial_guess=None):
    """Calculates the maximum-likelihood symmetric count matrix for a givnen observed count matrix.
    
    This function uses a Newton conjugate-gradient algorithm to maximize the likelihood
    of a reversible transition probability matrix.
    
    Parameters
    ----------
    count_matrix : array or sparse matrix
        Transition count matrix.
    prior : float
        If not zero, add this value to the count matrix for every transition
        that has occured in either direction.
    initial_guess : array or sparse matrix
        Initial guess for the symmetric count matrix uses as starting point for
        likelihood maximization. If None, the naive symmetrized guess 0.5*(C+C.T)
        is used.
        
    Returns
    -------
    reversible_counts : array or sparse matrix
        Symmetric count matrix. If C is sparse then the returned matrix is also sparse, and
        dense otherwise.
        
    """
    C = count_matrix
    
    def negativeLogLikelihoodFromCountEstimatesSparse(Xupdata, row, col, N, C):
        """Calculates the negative log likelihood that a symmetric count matrix X gave
    rise to an observed transition count matrix C, as well as the gradient
    d -log L / d X_ij."""
        
        assert np.alltrue(Xupdata > 0)        
        
        Xup = scipy.sparse.csr_matrix((Xupdata, (row, col)), shape=(N, N))                    # Xup is the upper triagonal (inluding the main diagonal) of the symmetric count matrix
        X = Xup + Xup.T - scipy.sparse.spdiags(Xup.diagonal(), 0, Xup.shape[0], Xup.shape[1])  # X is the complete symmetric count matrix
        Xs = np.array(X.sum(axis=1)).ravel()                                                # Xs is the array of row sums of X: Xs_i = sum_j X_ij
        XsInv = scipy.sparse.spdiags(1.0 / Xs, 0, len(Xs), len(Xs))
        P = (XsInv * X).tocsr()                                                             # P is now the matrix P_ij = X_ij / sum_j X_ij
        logP = scipy.sparse.csr_matrix((np.log(P.data), P.indices, P.indptr))
        logL = np.sum(C.multiply(logP).data)                                                # logL is the log of the likelihood: sum_ij C_ij log(X_ij / Xs_i)
        
        Cs = np.array(C.sum(axis=1)).ravel()                                                # Cs is the array of row sums of C: Cs_i = sum_j C_ij
        srow, scol = X.nonzero()                                                            # remember the postitions of the non-zero elements of X
        Udata = np.array((C[srow, scol] / X[srow, scol]) - (Cs[srow] / Xs[srow])).ravel()       # calculate the derivative: d(log L)/dX_ij = C_ij/X_ij - Cs_i/Xs_i
        U = scipy.sparse.csr_matrix((Udata, (srow, scol)), shape=(N, N))                        # U is the matrix U_ij = d(log L) / dX_ij
        
        # so far, we have assumed that all the partial derivatives wrt. X_ij are independent
        # however, the degrees of freedom are only X_ij for i <= j
        # for i != j, the total change in log L is d(log L)/dX_ij + d(log L)/dX_ji
        
        gradient = (U + U.T - scipy.sparse.spdiags(U.diagonal(), 0, U.shape[0], U.shape[1]) ).tocsr()
        
        # now we have to convert the non-zero elements of the upper triangle into the
        # same 1-d array structure that was used for Xupdata
        
        gradient = np.array(gradient[row, col]).reshape(-1)
        
        #print  "max g:", np.max(gradient), "min g:", np.min(gradient), "|g|^2", (gradient*gradient).sum(), "g * X", (gradient*Xupdata).sum()
        return -logL, -gradient
        
    # current implementation only for sparse matrices
    # if given a dense matrix, sparsify it, and turn the result back to a dense array
    if not scipy.sparse.isspmatrix(C):
        return mle_reversible_count_matrix(scipy.sparse.csr_matrix(C), prior=prior, initial_guess=initial_guess).toarray()
        
    N = C.shape[0]
    if not C.shape[1] == N:
        raise ValueError("Count matrix is not square, but has shape %s" % C.shape)
        
    C = C.tocsr()
    C.eliminate_zeros()
    # add prior if necessary
    if (prior is not None) and (prior != 0):
        PriorMatrix = (C+C.transpose()).tocsr()
        PriorMatrix.data *= 0.
        PriorMatrix.data += prior
        C = C + PriorMatrix
        logger.warning("Added prior value of %f to count matrix", prior)
        
    # initial guess for symmetric count matrix
    if initial_guess is None:
        X0 = 0.5 * (C + C.T)
    else:
        X0 = scipy.sparse.csr_matrix(0.5 * (initial_guess + initial_guess.T))  # this guarantees that the initial guess is indeed symmetric (and sparse)
    initialLogLikelihood = log_likelihood(C, estimate_transition_matrix(X0))
    
    # due to symmetry, we degrees of freedom for minimization are only the elments in the upper triangle of the matrix X (incl. main diagonal)
    X0up = scipy.sparse.triu(X0).tocoo()
    row = X0up.row
    col = X0up.col
    
    # the variables used during minimization are those X_ij (i <= j) for which either C_ij or C_ji is greater than zero
    # those X_ij can be arbitrariliy small, but they must be positive
    # the function minimizer requires an inclusive bound, so we use some very small number instead of zero
    # (without loss of generality, b/c we can always multiply all X_ij by some large number without changing the likelihood)
    lower_bound = 1.E-10
    bounds = [[lower_bound, np.inf]] * len(X0up.data)
    
    # Here comes the main loop
    # In principle, we would have to run the function minimizer only once. But in practice, minimization may fail
    # if the gradient term becomes too large, or minimization is slow if the gradient is too small.
    # Every so often, we therefore rescale the parameters X_ij so that the gradient is of resonable magnitude
    # (which does not affect the likelihood). This empirical procedure includes two parameters: the rescaling
    # frequency and the target value. In principles, these choices should not affect the outcome of the maximization.
    rescale_every = 500
    rescale_target = 1.
    
    Xupdata = X0up.data
    maximizationrun = 1
    totalnumberoffunctionevaluations = 0
    negative_logL, negative_gradient = negativeLogLikelihoodFromCountEstimatesSparse(Xupdata, row, col, N, C)
    logger.info("Log-Likelihood of intial guess for reversible transition probability matrix: %s", -negative_logL)
    while maximizationrun <= 1000:
        # rescale the X_ij so that the magnitude of the gradient is 1
        gtg = (negative_gradient*negative_gradient).sum()
        scalefactor = np.sqrt(gtg / rescale_target)
        Xupdata[:] *= scalefactor
        
        # now run the minimizer
        Xupdata, nfeval, rc = scipy.optimize.fmin_tnc(negativeLogLikelihoodFromCountEstimatesSparse,
                                        Xupdata, args=(row, col, N, C), bounds=bounds,
                                        approx_grad=False, maxfun=rescale_every, disp=0,
                                        xtol=1E-20)
                                        
        totalnumberoffunctionevaluations += nfeval
        negative_logL, negative_gradient = negativeLogLikelihoodFromCountEstimatesSparse(Xupdata, row, col, N, C)
        logger.info("Log-Likelihood after %s function evaluations; %s ", totalnumberoffunctionevaluations, -negative_logL)
        if rc in (0, 1, 2):
            break    # Converged
        elif rc in (3, 4):
            pass     # Not converged, keep going
        else:
            raise RuntimeError("Likelihood maximization caused internal error (code %s): %s" % (rc, scipy.optimize.tnc.RCSTRINGS[rc]))
        maximizationrun += 1
    else:
        logger.error("maximum could not be obtained.")
    logger.info("Result of last maximization run (run %s): %s", str(maximizationrun) , scipy.optimize.tnc.RCSTRINGS[rc])
    
    Xup = scipy.sparse.coo_matrix((Xupdata, (row, col)), shape=(N, N))    
    
    # reconstruct full symmetric matrix from upper triangle part
    X = Xup + Xup.T - scipy.sparse.spdiags(Xup.diagonal(), 0, Xup.shape[0], Xup.shape[1])
    
    finalLogLikelihood = log_likelihood(C, estimate_transition_matrix(X))
    logger.info("Log-Likelihood of final reversible transition probability matrix: %s", finalLogLikelihood)
    logger.info("Likelihood ratio: %s", np.exp(finalLogLikelihood - initialLogLikelihood))
    
    # some  basic consistency checks
    if not np.alltrue(np.isfinite(X.data)):
        raise RuntimeError("The obtained symmetrized count matrix is not finite.")
    if not np.alltrue(X.data > 0):
        raise RuntimeError("The obtained symmetrized count matrix is not strictly positive for all observed transitions, the smallest element is %s" % str(np.min(X.data)))
        
    #normalize X to have correct total number of counts
    X /= X.sum()
    X *= C.sum()
    
    return X


def permute_mat(A, permutation):
    """
    Permutes the indices of a transition probability matrix.
    
    This functions simply switches the lables of `A` rows and
    columns from [0, 1, 2, ...] to `permutation`.
    
    Parameters
    ----------
    tprob : matrix
    
    permutation: ndarray, int
        The permutation array, a list of unique indices that 
    
    Returns
    -------
    permuted_A : matrix
        The permuted matrix
    """
    
    if scipy.sparse.issparse(A):
        sparse = True
    else:
        sparse = False
    
    if sparse:
        
        Pi = scipy.sparse.lil_matrix(A.shape)
        for i in range(A.shape[0]):
            Pi[i,permutation[i]] = 1.0 # send i -> correct place
        permuted_A = Pi * A * Pi.T
        
    else:
        
        Pi = np.zeros(A.shape)
        for i in range(A.shape[0]):
            Pi[i,permutation[i]] = 1.0 # send i -> correct place
        permuted_A = np.dot( Pi, np.dot( A, Pi.T ) )
    
    return permuted_A


######################################################################
#  ALIASES FOR DEPRECATED FUNCTION NAMES
#  THESE FUNCTIONS WERE ADDED FOR VERSION 2.6 AND
#  CAN BE REMOVED IN VERSION 
######################################################################
@deprecated(msm_analysis.is_transition_matrix, '2.7')
def IsTransitionMatrix():
    pass
@deprecated(msm_analysis.are_all_dimensions_same, '2.7')
def AreAllDimensionsSame():
    pass
@deprecated(msm_analysis.check_dimensions, '2.7')
def CheckDimensions():
    pass
@deprecated(msm_analysis.check_transition, '2.7')
def CheckTransition():
    pass
@deprecated(get_counts_from_traj, '2.7')
def GetTransitionCountMatrixSparse():
    pass
@deprecated(estimate_rate_matrix, '2.7')
def EstimateRateMatrix():
    pass
@deprecated(estimate_transition_matrix, '2.7')
def EstimateTransitionMatrix():
    pass
@deprecated(msm_analysis.check_for_bad_eigenvalues, '2.7')
def CheckForBadEigenvalues():
    pass
@deprecated(msm_analysis.get_eigenvectors, '2.7')
def GetEigenvectors():
    pass
@deprecated(msm_analysis.get_implied_timescales, '2.7')
def GetImpliedTimescales():
    pass
@deprecated(get_count_matrix_from_assignments, '2.7')
def GetCountMatrixFromAssignments():
    pass
@deprecated(msm_analysis.sample, '2.7')
def Sample():
    pass
@deprecated(msm_analysis.propagate_model, '2.7')
def PropagateModel():
    pass
@deprecated(apply_mapping_to_assignments, '2.7')
def ApplyMappingToAssignments():
    pass
@deprecated(apply_mapping_to_vector, '2.7')
def ApplyMappingToVector():
    pass
@deprecated(ergodic_trim, '2.7')
def ErgodicTrim():
    pass
@deprecated(mle_reversible_count_matrix, '2.7')
def EstimateReversibleCountMatrix():
    pass
@deprecated(log_likelihood, '2.7')
def logLikelihood():
    pass
@deprecated(renumber_states, '2.7')
def RenumberStates():
    pass
def GetEigenvectors_Right(*args, **kwargs):
    warnings.warn('GetEigenvectors_Right is deprecated use get_eigenvectors() with the keyword Right=True')
    kwargs['right'] = True
    return msm_analysis.get_eigenvectors(*args, **kwargs)
