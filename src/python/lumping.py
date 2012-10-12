

import scipy.optimize
import scipy.io
import numpy as np

from msmbuilder import msm_analysis
from msmbuilder.utils import deprecated
import logging
logger = logging.getLogger(__name__)

from numpy import dot, diag
inv = np.linalg.inv
norm = np.linalg.norm
tr = np.trace


def normalize_left_eigenvectors(V):
    """Normalize the left eigenvectors
    
    Normalization condition is <v[:,i]/pi, v[:,i]> = 1
    
    Parameters
    ----------
    V : ndarray
        The left eigenvectors, as a two-dimensional array where the kth
        eigenvectors is V[:,k]
    
    Notes
    -----
    Acts inplace. Assumes that V[:,0] is the equilibrium vector and that detailed balance holds.
    """
    pi = V[:, 0]
    pi /= pi.sum()
    
    for k in xrange(1, V.shape[-1]):
        x = V[:, k]
        x /= abs(np.dot(x / pi, x)) ** .5


def trim_eigenvectors_by_flux(lam, vl, flux_cutoff):
    """
    Trim eigenvectors that have low equilibrium flux.

    Parameters
    ----------
    lam : nadarray
        Eigenvalues of transition matrix.
    vl : ndarray
        Left eigenvectors of transition matrix.
    flux_cutoff : float
        Discard eigenvectors with fluxes below this value.
    
    Notes
    -----
    Assuming that the left eigenvectors are properly pi-normalized,
    the equilibrium flux contribution of each eigenvector :math:`v` is given by :math:`\sum_i v_i^2`
    
    Returns
    -------
    lam : ndarray
        Eigenvalues after discarding low-flux eigenvectors.
    vl : ndarray
        Left eigenvectors after discarding low-flux eigenvectors.
    
    """
    normalize_left_eigenvectors(vl)

    N = len(lam)

    flux_list = np.array([(vl[:, i] ** 2).sum() for i in range(N)])
    flux_list /= flux_list[0]
    flux_list[0] = flux_list.max()
    
    KeepInd = np.where(flux_list >= flux_cutoff)[0]

    logger.info("Implied timescales (UNITLESS)")
    logger.info(-1 / np.log(lam))
    logger.info("Flux")
    logger.info(flux_list)
    logger.info("Keeping %d eigenvectors after flux cutoff %f", len(KeepInd), flux_cutoff)

    lam = lam[KeepInd]
    vl = vl[:, KeepInd]
    flux_list = flux_list[KeepInd]
    
    logger.info("After Flux calculation, Implied timescales (UNITLESS):")
    logger.info(-1 / np.log(lam))

    logger.info("After Flux calculation, fluxes.")
    logger.info(flux_list)

    return lam, vl


def get_maps(A):
    """Get mappings from the square array A to the flat vector of parameters alpha.
    
    Helper function for PCCA+ optimization.
    
    Parameters
    ----------
    A : ndarray
        The transformation matrix A.
    
    Returns
    -------
    flat_map : ndarray
        Mapping from flat indices (k) to square (i,j) indices.
    square map : ndarray
        Mapping from square indices (i,j) to flat indices (k).
    """

    N = A.shape[0]
    flat_map = []
    for i in range(1, N):
        for j in range(1, N):
            flat_map.append([i, j])

    flat_map = np.array(flat_map)

    square_map = np.zeros(A.shape, 'int')

    for k in range((N - 1) ** 2):
        i, j = flat_map[k]
        square_map[i, j] = k

    return flat_map, square_map


def to_flat(A, flat_map):
    """Convert a square matrix A to a flat array alpha.
    
    Parameters
    ----------
    A : ndarray
        The transformation matrix A
    flat_map : ndarray
        Mapping from flat indices (k) to square (i,j) indices.
    
    Returns
    -------
    FlatenedA : ndarray
        flattened version of A
    """
    return A[flat_map[:, 0], flat_map[:, 1]]


def to_square(alpha, square_map):
    """Convert a flat array alpha to a square array A.
    
    Parameters
    ----------
    alpha : ndarray
        An array of (n-1)^2 parameters used as optimization parameters.
        alpha is a minimal, flat representation of A.
    square_map : ndarray
        Mapping from square indices (i,j) to flat indices (k).
    Returns
    -------
    SquareA : ndarray
        Square version of alpha
    """
    return alpha[square_map]


def pcca_plus(T, N, flux_cutoff=None, do_minimization=True, objective_function="crisp_metastability"):
    """Perform PCCA+ lumping

    Parameters
    ----------
    T : csr sparse matrix
        Transition matrix
    M : int
        desired (maximum) number of macrostates
    flux_cutoff : float, optional
        If desired, discard eigenvectors with flux below this value.
    do_minimization : bool, optional
        If False, skip the optimization of the transformation matrix.
        In general, minimization is recommended.
    objective_function: {'crisp_metastablility', 'metastability', 'metastability'}
        Possible objective functions.  See objective for details.
    
    Returns
    -------
    A : ndarray
        The transformation matrix.
    chi : ndarray
        The membership matrix
    vr : ndarray
        The right eigenvectors.
    microstate_mapping : ndarray
        Mapping from microstates to macrostates.
    
    
    Notes
    -----
    PCCA+ is used to construct a "lumped" state decomposition.  First,
    The eigenvalues and eigenvectors are computed for a transition matrix.
    An optimization problem is then used to estimate a mapping from
    microstates to macrostates.
    
    For each microstate i, microstate_mapping[i] is chosen as the
    macrostate with the largest membership (chi) value.
    
    The membership matrix chi is given by chi = dot(vr,A).
    
    Finally, the transformation matrix A is the output of a constrained
    optimization problem.
    

    References
    ----------
    .. [1]  Deuflhard P, et al.  "Identification of almost invariant
    aggregates in reversible nearly uncoupled markov chains,"
    Linear Algebra Appl., vol 315 pp 39-59, 2000.

    .. [2]  Deuflhard P, Weber, M.,  "Robust perron cluster analysis in
     conformation dynamics,"
    Linear Algebra Appl., vol 398 pp 161-184 2005.
    
    .. [3]  Kube S, Weber M.  "A coarse graining method for the
    identification of transition rates between molecular conformations,"
    J. Chem. Phys., vol 126 pp 24103-024113, 2007.
    
    
    See Also
    --------
    PCCA
    """
    lam, vl = msm_analysis.get_eigenvectors(T, N)
    normalize_left_eigenvectors(vl)

    if flux_cutoff != None:
        lam, vl = trim_eigenvectors_by_flux(lam, vl, flux_cutoff)
        N = len(lam)
    
    pi = vl[:, 0]

    vr = vl.copy()
    for i in range(N):
        vr[:, i] /= pi
        vr[:, i] *= np.sign(vr[0, i])
        vr[:, i] /= np.sqrt(dot(vr[:, i] * pi, vr[:, i]))

    A, chi, microstate_mapping = opt_soft(vr, N, pi, lam, T, do_minimization=do_minimization, objective_function=objective_function)

    return A, chi, vr, microstate_mapping


def opt_soft(vr, N, pi, lam, T, do_minimization=True, objective_function="crisp_metastability"):
    """Perform PCCA+ algorithm by optimizing transformation matrix A.
    
  
    Parameters
    ----------
    vr :  ndarray
        Right eigenvectors of transition matrix
    N : int
        Desired number of macrostates
    pi: ndarray
        Equilibrium populations of transition matrix.
    lam : ndarray
        Eigenvalues of transition matrix
    T : csr sparse matrix
        Transition matrix
    do_minimzation : bool, optional
        Set to false if skipping minimization step.
    objective_function: {'crisp_metastablility', 'metastability', 'metastability'}
        Possible objective functions.  See objective for details.
    
    Returns
    -------
    A : ndarray
        The transformation matrix.
    chi : ndarray
        The membership matrix
    microstate_mapping : ndarray
        Mapping from microstates to macrostates.
    
    """

    index = index_search(vr)

    # compute transformation matrix A as initial guess for local optimization (maybe not feasible)
    A = vr[index, :]

    A = inv(A)
    A = fill_A(A, vr)

    if do_minimization == True:
        flat_map, square_map = get_maps(A)
        alpha = to_flat(1.0 * A, flat_map)

        obj = lambda x: -1 * objective(x, vr, square_map, lam, T, pi, objective_function=objective_function)

        logger.info("Initial value of objective function: %f", obj(alpha))

        alpha = scipy.optimize.anneal(obj, alpha, lower=0.0, maxiter=1, schedule="boltzmann", dwell=1000, feps=1E-3, boltzmann=2.0, T0=1.0)[0]

        alpha = scipy.optimize.fmin(obj, alpha, full_output=True, xtol=1E-4, ftol=1E-4, maxfun=5000, maxiter=100000)[0]

        logger.info("Final values.\n f = %f" % (-1 * obj(alpha)))

        A = to_square(alpha, square_map)

    else:
        logger.warning("Skipping Minimization")

    A = fill_A(A, vr)

    chi = dot(vr, A)

    microstate_mapping = np.argmax(chi, 1)

    return A, chi, microstate_mapping


def has_constraint_violation(A, vr, epsilon=1E-8):
    """Check for constraint violations in transformation matrix.
    
    Parameters
    ----------
    A : ndarray
        The transformation matrix.
    vr : ndarray
        The right eigenvectors.
    epsilon : float, optional
        Tolerance of constraint violation.
        
    Returns
    -------
    truth : bool
        Whether or not the violation exists
    
    
    Notes
    -------
    Checks constraints using Eqn 4.25 in [1].
    
    
    References
    ----------
    .. [1]  Deuflhard P, Weber, M.,  "Robust perron cluster analysis in
     conformation dynamics,"
    Linear Algebra Appl., vol 398 pp 161-184 2005.
    
    """

    lhs = 1 - A[0, 1:].sum()
    rhs = dot(vr[:, 1:], A[1:, 0])
    rhs = -1 * rhs.min()
    
    if abs(lhs - rhs) > epsilon:
        return True
    else:
        return False


def objective(alpha, vr, square_map, lam, T, pi, barrier_penalty=20000., objective_function="crisp_metastability"):
    """Return the PCCA+ objective function.

    Parameters
    ----------
    alpha : ndarray
        Parameters of objective function (e.g. flattened A)
    vr : ndarray
        The right eigenvectors.
    square_map : ndarray
        Mapping from square indices (i,j) to flat indices (k).
    lam : ndarray
        Eigenvalues of transition matrix.
    pi : ndarray
        Equilibrium Populations of transition matrix.
    objective_function: {'crisp_metastablility', 'metastability', 'metastability'}
        Possible objective functions.
        
        
    Returns
    -------
    obj : float
        The objective function
    
    
    Notes
    -------
    crispness: try to make crisp state decompostion.  This function is
    defined in [3].

    metastability: try to make metastable fuzzy state decomposition.
    Defined in ref. [2].

    crisp_metastability: try to make the resulting crisp msm metastable.
    This is the recommended choice.  This is the metastability (trace)
    of a transition matrix computed by forcing a crisp (non-fuzzy)
    microstate mapping.  Defined in ref. [2].
    
    References
    ----------
    .. [1]  Deuflhard P, Weber, M.,  "Robust perron cluster analysis in
     conformation dynamics,"
    Linear Algebra Appl., vol 398 pp 161-184 2005.
    
    .. [2]  Kube S, Weber M.  "A coarse graining method for the
    identification of transition rates between molecular conformations,"
    J. Chem. Phys., vol 126 pp 24103-024113, 2007.
    
    .. [3]  Kube S.,  "Statistical Error Estimation and Grid-free
    Hierarchical Refinement in Conformation Dynamics," Doctoral Thesis.
    2008
        
    """

    n, N = vr.shape

    A = to_square(alpha, square_map)

    #make A feasible
    A = fill_A(A, vr)

    chi_fuzzy = dot(vr, A)
    mapping = np.argmax(chi_fuzzy, 1)

    possible_objective_functions = ["crispness", "crisp_metastability", "metastability"]
    
    if objective_function not in possible_objective_functions:
        raise Exception("objective_function must be one of ", possible_objective_functions)
    
    if objective_function == "crisp_metastability":
        chi = 0.0 * chi_fuzzy
        chi[np.arange(n), mapping] = 1.
    elif objective_function == "metastability":
        chi = chi_fuzzy

    if objective_function in ["crisp_metastability", "metastability"]:
        #Calculate  metastabilty of the lumped model.  Eqn 4.20 in LAA.
        meta = 0.
        for i in range(N):
            meta += dot(T.dot(chi[:, i]), pi * chi[:, i]) / dot(chi[:, i], pi)
        
        obj = meta

    if objective_function == "crispness":
        #Calculate the crispness defined by Roeblitz Doctoral thesis.
        obj = tr(dot(diag(1. / A[0]), dot(A.transpose(), A)))

    """
    If microstate_mapping is degenerate (not enough macrostates), we increase obj
    This prevents PCCA+ returning a model with insufficiently many states.
    We also do this if the macrostate populations are too low.
    """

    if len(np.unique(mapping)) != N or has_constraint_violation(A, vr):
        logger.warning("Constraint violation detected.")
        obj -= barrier_penalty

    logger.info("f = %f", obj.real)
    
    return obj


def fill_A(A, vr):
    """Construct feasible initial guess for transformation matrix A.
    
  
    Parameters
    ----------
    A : ndarray
        Possibly non-feasible transformation matrix.
    vr :  ndarray
        Right eigenvectors of transition matrix
    
    Returns
    -------
    A : ndarray
        Feasible transformation matrix.
    
    """
    n, N = vr.shape
    
    A = A.copy()
    
    #compute 1st column of A by row sum condition
    A[1:, 0] = -1 * A[1:, 1:].sum(1)

    # compute 1st row of A by maximum condition
    A[0] = -1 * dot(vr[:, 1:].real, A[1:]).min(0)

    #rescale A to be in the feasible set
    A /= A[0].sum()

    return A


def index_search(vr):
    """Find simplex structure in eigenvectors to begin PCCA+.
    
  
    Parameters
    ----------
    vr :  ndarray
        Right eigenvectors of transition matrix
    
    Returns
    -------
    index : ndarray
        Indices of simplex
    
    """

    n, N = vr.shape

    index = np.zeros(N, 'int')
    
    # first vertex: row with largest norm
    index[0] = np.argmax([norm(vr[i]) for i in range(n)])

    OrthoSys = vr - np.outer(np.ones(n), vr[index[0]])

    for j in range(1, N):
        temp = OrthoSys[index[j - 1]].copy()
        for l in range(n):
            OrthoSys[l] -= temp * dot(OrthoSys[l], temp)

        distlist = np.array([norm(OrthoSys[l]) for l in range(n)])

        index[j] = np.argmax(distlist)

        OrthoSys /= distlist.max()

    return index


def PCCA(T, num_macro, tolerance=1E-5, flux_cutoff=None):
    """Create a lumped model using the PCCA algorithm.
    
    1.  Iterate over the eigenvectors, starting with the slowest.
    2.  Calculate the spread of that eigenvector within each existing macrostate.
    3.  Pick the macrostate with the largest eigenvector spread.
    4.  Split the macrostate based on the sign of the eigenvector.

    Parameters
    ----------
    T : csr sparse matrix
        A transition matrix
    num_macro : int
        The desired number of states.
    tolerance : float, optional
        Specifies the numerical cutoff to use when splitting states based on sign.
    flux_cutoff : float, optional
        If enabled, discard eigenvectors with flux below this value.

    Returns
    -------
    microstate_mapping : ndarray
        mapping from the Microstate indices to the Macrostate indices

    Notes
    -----
    To construct a Macrostate MSM, you then need to map your Assignment data to
    the new states (e.g. MSMLib.apply_mapping_to_assignments).

    References
    ----------
    .. [1]  Deuflhard P, et al.  "Identification of almost invariant
    aggregates in reversible nearly uncoupled markov chains,"
    Linear Algebra Appl., vol 315 pp 39-59, 2000.

    """
    logger.info("in PCCA")

    n = T.shape[0]
    lam, vl = msm_analysis.get_eigenvectors(T, num_macro)
    normalize_left_eigenvectors(vl)

    if flux_cutoff is not None:
        lam, vl = trim_eigenvectors_by_flux(lam, vl, flux_cutoff)
        num_macro = len(lam)
    
    pi = vl[:, 0]

    vr = vl.copy()
    for i in range(num_macro):
        vr[:, i] /= pi
        vr[:, i] *= np.sign(vr[0, i])
        vr[:, i] /= np.sqrt(dot(vr[:, i] * pi, vr[:, i]))

    #Remove the stationary eigenvalue and eigenvector.
    lam = lam[1:]
    vr = vr[:, 1:]

    microstate_mapping = np.zeros(n, 'int')

    #Function to calculate the spread of a single eigenvector.
    spread = lambda x: x.max() - x.min()
    """
    1.  Iterate over the eigenvectors, starting with the slowest.
    2.  Calculate the spread of that eigenvector within each existing macrostate.
    3.  Pick the macrostate with the largest eigenvector spread.
    4.  Split the macrostate based on the sign of the eigenvector.
    """
    
    for i in range(num_macro - 1):  # Thus, if we want 2 states, we split once.
        v = vr[:, i]
        AllSpreads = np.array([spread(v[microstate_mapping == k]) for k in range(i + 1)])
        StateToBeSplit = np.argmax(AllSpreads)
        microstate_mapping[(microstate_mapping == StateToBeSplit) & (v >= tolerance)] = i + 1

    return(microstate_mapping)


@deprecated(normalize_left_eigenvectors, '2.7')
def NormalizeLeftEigenvectors():
    pass
