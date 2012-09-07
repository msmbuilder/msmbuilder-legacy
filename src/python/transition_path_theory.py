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

"""
Functions for performing Transition Path Theory calculations.

Written and maintained by TJ Lane <tjlane@stanford.edu>
Contributions from Kyle Beauchamp, Robert McGibbon, Vince Voelz

To Do:
-- Get stuff passing unit tests
-- Finish renaming functions
-- Add logging functionality

"""

import numpy as np
import scipy.sparse

from msmbuilder import MSMLib
from msmbuilder import msm_analysis
from msmbuilder.utils import deprecated

import logging
logger = logging.getLogger('tpt')

# turn on debugging printout
# logger.setLogLevel(logging.DEBUG)



###############################################################################
# Typechecking/Utility Functions
#

def _ensure_iterable(arg):
    if not hasattr(arg, '__iter__'):
        arg = list([int(arg)])
        print("Warning: passed object was not iterable,"
              " converted it to: %s" % str(arg))
    return arg

def _check_sources_sinks(sources, sinks):
    sources = _ensure_iterable(sources)
    sinks = _ensure_iterable(sinks)
    if np.any( sources == sinks ):
        raise ValueError("Sets `sources` and `sinks` must be disjoint "
                         "to find paths between them")
    return sources, sinks


###############################################################################
# Path Finding Functions
#

def DijkstraTopPaths(sources, sinks, net_flux, num_paths=10, node_wipe=False):
    r"""
    Calls the Dijkstra algorithm to find the top 'NumPaths'. 

    Does this recursively by first finding the top flux path, then cutting that
    path and relaxing to find the second top path. Continues until NumPaths
    have been found.

    Parameters
    ----------
    sources : array_like, int 
        The indices of the source states
    sinks : array_like, int
        Indices of sink states
    net_flux : sparse matrix
        Matrix of the net flux from `sources` to `sinks`, see function compute_net_flux
    num_paths : int 
        The number of paths to find
    node_wipe : bool 
        If true, removes the bottleneck-generating node from the graph, instead
        of just the bottleneck (not recommended, a debugging functionality)

    Returns
    -------
    Paths : list of lists
        The nodes transversed in each path
    Bottlenecks : list of tuples
        The nodes between which exists the path bottleneck
    Fluxes : a list of floats
        The flux through each path

    To Do
    -----
    -- Add periodic flow check
    """

    # first, do some checking on the input, esp. `sources` and `sinks`
    # we want to make sure all objects are iterable and the sets are disjoint
    
    sources, sinks = _check_sources_sinks(sources, sinks)

    # initialize objects
    Paths = []
    Fluxes = []
    Bottlenecks = []
    net_flux = net_flux.tolil()

    # run the initial Dijkstra pass
    pi, b = Dijkstra(sources, sinks, net_flux)

    logger.info("Path Num | Path | Bottleneck | Flux")

    i = 1
    done = False
    while not done:

        # First find the highest flux pathway
        (Path, (b1,b2), Flux) = Backtrack(sinks, b, pi, net_flux)

        # Add each result to a Paths, Bottlenecks, Fluxes list
        if Flux == 0:
            logger.info("Only %d possible pathways found. Stopping backtrack.", i)
            break
        Paths.append(Path)
        Bottlenecks.append( (b1,b2) )
        Fluxes.append(Flux)
        logger.info("%s | %s | %s | %s ", i, Path, (b1, b2), Flux)

        # Cut the bottleneck, start relaxing from B side of the cut
        if NodeWipe: 

            net_flux[:, b2] = 0
            logger.info("Wiped node: %s", b2)
        else: net_flux[b1, b2] = 0

        G = scipy.sparse.find(net_flux)
        Q = [b2]
        b, pi, NFlux = BackRelax(b2, b, pi, net_flux)
        
        # Then relax the graph and repeat
        # But only if we still need to
        if i != NumPaths-1:
            while len(Q) > 0:
                w = Q.pop()
                for v in G[1][np.where( G[0] == w )]:
                    if pi[v] == w:
                        b, pi, NFlux = BackRelax(v, b, pi, net_flux)
                        Q.append(v)
                Q = sorted(Q, key=lambda v: b[v])

        i+=1
        if i == NumPaths+1: 
            done = True
        if Flux == 0: 
            logger.info("Only %d possible pathways found. Stopping backtrack.", i)
            done = True

    return Paths, Bottlenecks, Fluxes


def Dijkstra(A, B, NFlux):
    r""" A modified Dijkstra algorithm that dynamically computes the cost
    of all paths from A to B, weighted by NFlux.

    Parameters
    ----------
    A : array_like, int 
        The indices of the source states (i.e. for state A in rxn A -> B)
    B : array_like, int
        Indices of sink states (state B)
    NFlux : sparse matrix
        Matrix of the net flux from A to B, see function GetFlux

    Returns
    -------
    pi : array_like
        The paths from A->B, pi[i] = node preceeding i
    b : array_like
        The flux passing through each node

    See Also
    --------
    DijkstraTopPaths : child function
        `DijkstraTopPaths` is probably the function you want to call to find
         paths through an MSM network. This is a utility function called by
         `DijkstraTopPaths`, but may be useful in some specific cases
    """

    # Initialize
    NFlux = NFlux.tolil()
    G = scipy.sparse.find(NFlux)
    N = NFlux.shape[0]
    b = np.zeros(N)
    b[A] = 1000
    pi = np.zeros(N, dtype=int)
    pi[A] = -1
    U = []
    Q = sorted(range(N), key=lambda v: b[v])
    for v in B: Q.remove(v)

    # Run
    while len(Q) > 0:
        w = Q.pop()
        U.append(w)
        
        # Relax
        for v in G[1][np.where( G[0] == w )]:
            if b[v] < min(b[w], NFlux[w,v]):
                b[v] = min(b[w], NFlux[w,v])
                pi[v] = w

        Q = sorted(Q, key=lambda v: b[v])

    logger.info("Searched %s nodes", len(U)+len(B))

    return pi, b


def BackRelax(s, b, pi, NFlux):
    r"""
    Updates a Djikstra calculation once a bottleneck is cut, quickly
    recalculating only cost of nodes that change due to the cut.

    Cuts & relaxes the B-side (sink side) of a cut edge (b2) to source from the
    adjacent node with the most flux flowing to it. If there are no
    adjacent source nodes, cuts the node out of the graph and relaxes the
    nodes that were getting fed by b2 (the cut node).

    Parameters
    ----------
    s : int
        the node b2
    b : array_like
        the cost function
    pi : array_like
        the backtrack array, a list such that pi[i] = source node of node i
    NFlux : sparse matrix
        Net flux matrix

    Returns
    -------
    b : array_like
        updated cost function
    pi : array_like
        updated backtrack array
    NFlux : sparse matrix
        net flux matrix

    See Also
    --------
    DijkstraTopPaths : child function
        `DijkstraTopPaths` is probably the function you want to call to find
         paths through an MSM network. This is a utility function called by
         `DijkstraTopPaths`, but may be useful in some specific cases
    """

    G = scipy.sparse.find(NFlux)
    if len( G[0][np.where( G[1] == s )] ) > 0:

        # For all nodes connected upstream to the node `s` in question,
        # Re-source that node from the best option (lowest cost) one level lower
        # Notation: j is node one level below, s is the one being considered

        b[s] = 0                                 # set the cost to zero
        for j in G[0][np.where( G[1] == s )]:    # for each upstream node
            if b[s] < min( b[j], NFlux[j,s] ):   # if that node has a lower cost
                b[s] = min( b[j], NFlux[j,s] )   # then set the cost to that node
                pi[s] = j                        # and the source comes from there

    # if there are no nodes connected to this one, then we need to go one
    # level up and work there first 
    else: 
        for sprime in G[1][np.where( G[0] == s )]:
            NFlux[s,sprime] = 0
            b, pi, NFlux = BackRelax(sprime, b, pi, NFlux)
            
    return b, pi, NFlux


def Backtrack(B, b, pi, NFlux):
    """
    Works backwards to pull out a path from pi, where pi is a list such that
    pi[i] = source node of node i. Begins at the largest staring incoming flux
    point in B.

    Parameters
    ----------
    B : array_like, int
        Indices of sink states (state B)
    b : array_like
        the cost function
    pi : array_like
        the backtrack array, a list such that pi[i] = source node of node i
    NFlux : sparse matrix
        net flux matrix
    
    Returns
    -------
    bestpath : list
        the list of nodes forming the highest flux path
    bottleneck : tuple
        a tupe of nodes, between which is the bottleneck
    bestflux : float
        the flux through `bestpath` 

    See Also
    --------
    DijkstraTopPaths : child function
        `DijkstraTopPaths` is probably the function you want to call to find
        paths through an MSM network. This is a utility function called by
        `DijkstraTopPaths`, but may be useful in some specific cases
    """

    # Select starting location
    bestflux = 0
    for Bnode in B:
        path = [Bnode]
        NotDone=True
        while NotDone:
            if pi[path[-1]] == -1:
                break
            else:
                #print pi
                path.append( pi[path[-1]] )
                #print path # sys.exit()
        path.reverse()

        bottleneck, Flux = FindPathBottleneck(path, NFlux)

        logger.debug('In Backtrack: Flux %s, bestflux %s', Flux, bestflux)
        
        if Flux > bestflux: 
            bestpath = path
            bestflux = Flux

    if Flux == 0:
        bestpath = []
        bottleneck = (np.nan, np.nan)
        bestflux = 0

    return (bestpath, bottleneck, bestflux)


def FindPathBottleneck(Path, NFlux):
    """
    Simply finds the bottleneck along a dynamically generated path. 
    This is the point at which the cost function first goes up along the path,
    backtracking from B to A.

    Parameters
    ----------
    Path : list
        a list of nodes along the path of interest
    NFlux : sparse matrix
        the net flux matrix

    Returns
    -------
    bottleneck : tuple
        a tuple of the nodes on either end of the bottleneck
    flux : float
        the flux at the bottleneck

    See Also
    --------
    DijkstraTopPaths : child function
        `DijkstraTopPaths` is probably the function you want to call to find
         paths through an MSM network. This is a utility function called by
         `DijkstraTopPaths`, but may be useful in some specific cases
    """

    NFlux = NFlux.tolil()
    flux = 100.

    for i in range(len(Path)-1):
        if NFlux[ Path[i], Path[i+1] ] < flux:
            flux = NFlux[ Path[i], Path[i+1] ]
            b1 = Path[i]
            b2 = Path[i+1]

    return (b1, b2), flux


def compute_fluxes(sources, sinks, tprob, populations=None, committors=None):
    """
    Compute the transition path theory flux matrix.
	
    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.

    sinks : array_like, int
        The set of folded/product states.
		
    tprob : mm_matrix	
        The transition matrix.
		
		
    Returns
    ------
    fluxes : mm_matrix
        The flux matrix
        
        
    Optional Parameters
    -------------------
    populations : nd_array, float
        The equilibrium populations, if not provided is re-calculated
        
    committors : nd_array, float
        The committors associated with `sources`, `sinks`, and `tprob`.
        If not provided, is calculated from scratch. If provided, `sources`
        and `sinks` are ignored.
    """
    
    sources, sinks = _check_sources_sinks(sources, sinks)
    
    # check if we got the populations
    if populations is None:
        eigens = msm_analysis.get_eigenvectors(tprob, 5)
        if np.count_nonzero(np.imag(eigens[1][:,0])) != 0:
            raise ValueError('First eigenvector has imaginary parts')
        populations = np.real(eigens[1][:,0])
        
    # check if we got the committors
    if committors is None:
        committors = calc_committors(sources, sinks, tprob, dense=False)
    
    # perform the flux computation
    Indx, Indy = tprob.nonzero()

    n = tprob.shape[0]
    X = scipy.sparse.lil_matrix( (n,n) )
    X.setdiag( populations * (1.0 - committors) )

    Y = scipy.sparse.lil_matrix( (n,n) )
    Y.setdiag(committors)
    fluxes = np.dot( np.dot(X.tocsr(), tprob.tocsr()), Y.tocsr() )
    fluxes = P.tolil()
    fluxes.setdiag(np.zeros(n))
    
    return fluxes


def compute_net_fluxes(sources, sinks, tprob, populations=None, committors=None):
    """
    Computes the transition path theory net flux matrix.

    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.

    sinks : array_like, int
        The set of folded/product states.
		
    tprob : mm_matrix	
        The transition matrix.
		
		
    Returns
    ------
    net_fluxes : mm_matrix
        The net flux matrix
        
        
    Optional Parameters
    -------------------
    populations : nd_array, float
        The equilibrium populations, if not provided is re-calculated
        
    committors : nd_array, float
        The committors associated with `sources`, `sinks`, and `tprob`.
        If not provided, is calculated from scratch. If provided, `sources`
        and `sinks` are ignored.
    """

    sources, sinks = _check_sources_sinks(sources, sinks)

    n = tprob.shape[0]
    
    flux = compute_net_fluxes(sources, sinks, tprob, populations, committors)
    ind = flux.nonzero()
    
    net_flux = scipy.sparse.lil_matrix( (n,n) )
    for k in range( len(ind[0]) ):
        i,j = ind[0][k], ind[1][k]
        forward = flux[i,j]
        reverse = flux[j,i]
        net_flux[i,j] = max(0, forward - reverse)
        
    return net_flux


###############################################################################
# MFPT & Committor Finding Functions
#

def calc_ensemble_mfpt(sources, sinks, tprob, lag_time):
    """
    Calculates the average 'Folding Time' of an MSM defined by T and a LagTime.
    The Folding Time is the average of the MFPTs (to F) of all the states in U.

    Note here 'Folding Time' is defined as the avg MFPT of {U}, to {F}.
    Consider this carefully. This is probably NOT the experimental folding time!

    Parameters
    ----------
    sources : array, int
        indices of the source states
    sinks : array, int 
        indices of the sink states
    tprob : matrix
        transition probability matrix
    lag_time : float
        the lag time used to create T (dictates units of the answer)

    Returns
    -------
    avg : float
        the average of the MFPTs
    std : float
        the standard deviation of the MFPTs
    """
    
    sources, sinks = _check_sources_sinks(sources, sinks)

    X = mfpt(sinks, tprob, lag_time)
    times = np.zeros(len(sources))
    for i in range(len(sources)):
        times[i] = X[ sources[i] ] 

    return np.average(times), np.std(times) 


def calc_avg_TP_time(sources, sinks, tprob, lag_time):
    """
    Calculates the Average Transition Path Time for MSM with: T, LagTime.
    The TPTime is the average of the MFPTs (to F) of all the states
    immediately adjacent to U, with the U states effectively deleted.

    Note here 'TP Time' is defined as the avg MFPT of all adjacent states to {U},
    to {F}, ignoring {U}.

    Consider this carefully.

    Parameters
    ----------
    sources : array, int
        indices of the unfolded states
    sinks : array, int 
        indices of the folded states
    tprob : matrix
        transition probability matrix
    lag_time : float
        the lag time used to create T (dictates units of the answer)

    Returns
    -------
    avg : float
        the average of the MFPTs
    std : float
        the standard deviation of the MFPTs
    """
    
    sources, sinks = _check_sources_sinks(sources, sinks)

    T = T.tolil()
    n = T.shape[0]
    P = scipy.sparse.lil_matrix((n,n))

    for u in sources:
        for i in range(n):
            if i not in U:
                P[u,i]=T[u,i]

    for u in sources:
        T[u,:] = np.zeros(n)
        T[:,u] = 0

    for i in sources:
        N = T[i,:].sum()
        T[i,:] = T[i,:]/N

    X = mfpt(sinks, prob, lag_time)
    TP = P * X.T
    TPtimes = []

    for time in TP:
        if time != 0: TPtimes.append(time)

    return np.average(TPtimes), np.std(TPtimes)


def mfpt(sinks, tprob, lag_time=1.):
    """
    Gets the Mean First Passage Time (MFPT) for all states to a *set*
    of sinks.

    Parameters
    ----------
    sinks : array, int 
        indices of the sink states
    tprob : matrix
        transition probability matrix
    LagTime : float
        the lag time used to create T (dictates units of the answer)

    Returns
    -------
    MFPT : array, float
        MFPT in time units of LagTime, for each state (in order of state index)

    See Also
    --------
    all_to_all_mfpt : function
        A more efficient way to calculate all the MFPTs in a network
    """
    
    sinks = _ensure_iterable(sinks)

    n = tprob.shape[0]
    
    if scipy.sparse.isspmatrix(T):
        tprob = tprob.tolil()
    
    for state in sinks:
        tprob[state,:] = 0.0    
        tprob[state,state] = 2.0

    if scipy.sparse.isspmatrix(T):
        tprob = tprob - scipy.sparse.eye(n,n)
        tprob = tprob.tocsr()
    else:
        tprob = tprob - np.eye(n)

    RHS = -1 * np.ones(n)
    for state in sinks:
        RHS[state] = 0.0

    if scipy.sparse.isspmatrix(tprob):
        MFPT = lag_time * scipy.sparse.linalg.spsolve(tprob, RHS)
    else:
        MFPT = lag_time * np.linalg.solve(tprob, RHS)

    return MFPT


def all_to_all_mfpt(tprob, populations=None):
    """
    Calculate the all-states by all-state matrix of mean first passage
    times.

    This uses the fundamental matrix formalism, and should be much faster
    than GetMFPT for calculating many MFPTs.

    Parameters
    ----------
    tprob : matrix
        transition probability matrix
    populations : array_like, float
        optional argument, the populations of each state. If  not supplied, 
        it will be computed from scratch

    Returns
    -------
    MFPT : array, float
        MFPT in time units of LagTime, square array for MFPT from i -> j

    See Also
    --------
    GetMFPT : function
        for calculating a subset of the MFPTs, with functionality for including
        a set of sinks
    """

    if populations is None:
        eigens = msm_analysis.get_eigenvectors(tprob, 5)
        if np.count_nonzero(np.imag(eigens[1][:,0])) != 0:
            raise ValueError('First eigenvector has imaginary parts')
        populations = np.real(eigens[1][:,0])

    # ensure that tprob is a transition matrix
    msm_analysis.check_transition(tprob)
    num_states = len(populations)
    if tprob.shape[0] != num_states:
        raise ValueError("Shape of tprob and populations vector don't match")

    eye = np.matrix(np.ones(num_states)).transpose()
    limiting_matrix = eye * populations
    z = scipy.linalg.inv(scipy.sparse.eye(num_states, num_states) - (tprob - limiting_matrix))

    # mfpt[i,j] = z[j,j] - z[i,j] / pi[j]
    mfpt = -z
    for j in range(num_states):
        mfpt[:, j] += z[j,j]
        mfpt[:, j] /= populations[j]

    return mfpt


def calc_committors(sources, sinks, tprob):
    """
    Get the forward committors of the reaction U -> F.

    If you are have small matrices, it can be faster to use dense 
    linear algebra. 
	
    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.

    sinks : array_like, int
        The set of folded/product states.
		
    tprob : mm_matrix	
        The transition matrix.
			
    dense : bool
        Employ dense linear algebra. Will speed up the calculation
        for small matrices.
		
    Returns
    -------
    Q : array_like
        The forward committors for the reaction U -> F.
    """
	
    sources, sinks = _check_sources_sinks(sources, sinks)

    if scipy.sparse.issparse(tprob):
        dense = True
    else:
        dense = False

    # construct the committor problem
    n = tprob.shape[0]

    if dense:
       T = np.eye(n) - tprob
    else:
       T = scipy.sparse.eye(n, n, 0, format='lil') - tprob

    for a in sources:
        T[a,:] = 0.0 #np.zeros(n)
        T[:,a] = 0.0
        T[a,a] = 1.0
        
    for b in sinks:
        T[b,:] = 0.0 # np.zeros(n)
        T[:,b] = 0.0
        T[b,b] = 1.0
        
    IdB = np.zeros(n)
    IdB[sinks] = 1.0
        
    RHS = tprob * IdB
    
    RHS[sources] = 0.0
    RHS[sinks]   = 1.0

    # solve for the committors
    if dense == False:
        Q = scipy.sparse.linalg.spsolve(T, RHS)
    else:
        Q = np.linalg.solve(T, RHS)

    return Q
    

######################################################################
#  DEPRECATED FUNCTIONS
#  CAN BE REMOVED IN VERSION 2.7
######################################################################

# TJL: Unnecessary helper function
@deprecated(calc_committors, '2.7')
def GetFCommittorsEqn(A, B, T0, dense=False):
    """
    Construct the matrix equations used for finding committors 
    for the reaction U -> F.  T0 is the transition matrix, 
    Equilibruim is the vector of equilibruim populations.
	
    Parameters
    ----------
    A : array_like, int
        The set of unfolded/reactant states.

    B : array_like, int
        The set of folded/product states.
		
    T0 : mm_matrix	
        The transition matrix.
				
    Returns
    -------
    A : mm_matrix
        the sparse matrix for committors (Ax = b)
 
    b : array_like
        the vector on right hand side of (Ax = b)
    """
	
    # TJL to KAB : please be consistent with variable names!
	
    n = T0.shape[0]

    if dense:
       T = np.linalg.eye(n) - T0

    else:
       T = scipy.sparse.eye(n,n,0,format='lil')-T0
       T = T.tolil()

    for a in A:
        T[a,:] = np.zeros(n)
        T[:,a] = 0.0
        T[a,a] = 1.0
    for b in B:
        T[b,:] = np.zeros(n)
        T[:,b] = 0.0
        T[b,b] = 1.0
    IdB = np.zeros(n)
    for b in B:
        IdB[b] = 1.0
    logger.info("done with setting up matrices")
    RHS = T0*(IdB) # changed from RHS=T0.matvec(IdB) --TJL, matvec deprecated
    for a in A:
        RHS[a] = 0.0
    for b in B:
        RHS[b] = 1.0

    return (T,RHS)


# TJL: these helper functions are cluttered/confusing
@deprecated(calc_committors, '2.7')
def GetBCommittorsEqn(U, F, T0, EquilibriumPopulations):
    """
    Construct the matrix equations used for finding backwards 
    committors for the reaction U -> F.
	
    Parameters
    ----------
    U : array_like, int
        The set of unfolded/reactant states.
		
    F : array_like, int
        The set of folded/product states.
		
    T0 : mm_matrix	
        The transition matrix.
		
    EquilibriumPopulations : array_like, float
        Populations of the states.
		
    Returns
    -------
     A : mm_matrix 
        the sparse matrix for committors (Ax = b)
 
     b : nd_array
         the vector on right hand side of (Ax = b)
    """

    n = len(EquilibriumPopulations)
    DE = scipy.sparse.eye(n,n,0,format='lil')
    DE.setdiag(EquilibriumPopulations)
    DEInv = scipy.sparse.eye(n,n,0,format='lil')
    DEInv.setdiag(1./EquilibriumPopulations)
    TR = (DEInv.dot(T0.transpose())).dot(DE)

    return GetFCommittorsEqn(F,U,TR)


# TJL: We don't need a "get backwards committors", since they are just
# 1 minus the forward committors
@deprecated(calc_committors, '2.7')
def GetBCommittors(U, F, T0, EquilibriumPopulations, X0=None, Dense=False):
    """
    Get the backward committors of the reaction U -> F.
	
    EquilibriumPopulations are required for the backward committors 
    but not the foward commitors because we assume detailed balance 
    when calculating the backward committors.  If you are have small 
    matrices, it can be faster to use dense linear algebra.  
	
    Parameters
    ----------
    U : array_like, int
        The set of unfolded/reactant states.
		
    F : array_like, int
        The set of folded/product states.
		
    T0 : mm_matrix	
        The transition matrix.
		
    EquilibriumPopulations : array_like, float
        Populations of the states.
						
    Dense : bool
        Employ dense linear algebra. Will speed up the calculation
        for small matrices.
		
    Returns
    -------
    Q : array_like
        The backward committors for the reaction U -> F.	
    """
	
    A, b = GetBCommittorsEqn(U,F,T0,EquilibriumPopulations)

    if Dense==False:
        Q = scipy.sparse.linalg.spsolve(A,b)
    else:
        Q = np.linalg.solve(A.toarray(),b)

    return Q


######################################################################
#  ALIASES FOR DEPRECATED FUNCTION NAMES
#  THESE FUNCTIONS WERE ADDED FOR VERSION 2.6 AND
#  CAN BE REMOVED IN VERSION 2.7
######################################################################
@deprecated(mfpt, '2.7')
def MFPT():
    pass

@deprecated(calc_committors, '2.7')
def GetFCommittors():
    pass
    
@deprecated(compute_fluxes, '2.7')
def GetFlux():
    print "WARNING: The call signature for the new function has changed."
    pass

@deprecated(compute_net_fluxes, '2.7')
def GetNetFlux():
    print "WARNING: The call signature for the new function has changed."
    pass
    
@deprecated(calc_avg_TP_time, '2.7')
def CalcAvgTPTime():
    pass
    
@deprecated(calc_ensemble_mfpt, '2.7')
def CalcAvgFoldingTime():
    pass
    
