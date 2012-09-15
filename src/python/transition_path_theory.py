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
        logger.debug("Passed object was not iterable,"
                     " converted it to: %s" % str(arg))
    assert hasattr(sources, '__iter__')
    assert hasattr(sinks, '__iter__')
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

def find_top_paths(sources, sinks, tprob, num_paths=10, node_wipe=False, net_flux=None):
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
    num_paths : int 
        The number of paths to find

    Returns
    -------
    Paths : list of lists
        The nodes transversed in each path
    Bottlenecks : list of tuples
        The nodes between which exists the path bottleneck
    Fluxes : list of floats
        The flux through each path
        
    Optional Parameters
    -------------------
    node_wipe : bool 
        If true, removes the bottleneck-generating node from the graph, instead
        of just the bottleneck (not recommended, a debugging functionality)
    net_flux : sparse matrix
        Matrix of the net flux from `sources` to `sinks`, see function `net_flux`.
        If not provided, is calculated from scratch. If provided, `tprob` is
        ignored.

    To Do
    -----
    -- Add periodic flow check
    """

    # first, do some checking on the input, esp. `sources` and `sinks`
    # we want to make sure all objects are iterable and the sets are disjoint
    sources, sinks = _check_sources_sinks(sources, sinks)
    
    # check to see if we get net_flux for free, otherwise calculate it
    if not net_flux:
        net_flux = calculate_net_fluxes(sources, sinks, tprob)

    # initialize objects
    paths = []
    fluxes = []
    bottlenecks = []
    
    if scipy.sparse.issparse(net_flux):
        net_flux = net_flux.tolil()

    # run the initial Dijkstra pass
    pi, b = Dijkstra(sources, sinks, net_flux)

    logger.info("Path Num | Path | Bottleneck | Flux")

    i = 1
    done = False
    while not done:

        # First find the highest flux pathway
        (path, (b1,b2), flux) = _backtrack(sinks, b, pi, net_flux)

        # Add each result to a Paths, Bottlenecks, Fluxes list
        if flux == 0:
            logger.info("Only %d possible pathways found. Stopping backtrack.", i)
            break
        paths.append(path)
        bottlenecks.append( (b1,b2) )
        fluxes.append(flux)
        logger.info("%s | %s | %s | %s ", i, path, (b1, b2), flux)

        # Cut the bottleneck, start relaxing from B side of the cut
        if node_wipe: 
            net_flux[:, b2] = 0
            logger.info("Wiped node: %s", b2)
        else:
            net_flux[b1, b2] = 0

        G = scipy.sparse.find(net_flux)
        Q = [b2]
        b, pi, net_flux = _back_relax(b2, b, pi, net_flux)
        
        # Then relax the graph and repeat
        # But only if we still need to
        if i != num_paths - 1:
            while len(Q) > 0:
                w = Q.pop()
                for v in G[1][np.where( G[0] == w )]:
                    if pi[v] == w:
                        b, pi, net_flux = _back_relax(v, b, pi, net_flux)
                        Q.append(v)
                Q = sorted(Q, key=lambda v: b[v])

        i += 1
        if i == num_paths + 1: 
            done = True
        if flux == 0: 
            logger.info("Only %d possible pathways found. Stopping backtrack.", i)
            done = True

    return paths, bottlenecks, fluxes


def Dijkstra(sources, sinks, net_flux):
    r""" A modified Dijkstra algorithm that dynamically computes the cost
    of all paths from A to B, weighted by NFlux.

    Parameters
    ----------
    sources : array_like, int 
        The indices of the source states (i.e. for state A in rxn A -> B)
    sinks : array_like, int
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
    
    sources, sinks = _check_sources_sinks(sources, sinks)

    # initialize data structures
    if scipy.sparse.issparse(net_flux):
        net_flux = net_flux.tolil()
    else:
        net_flux = scipy.sparse.lil_matrix(net_flux)
        
    G = scipy.sparse.find(net_flux)
    N = net_flux.shape[0]
    b = np.zeros(N)
    b[sources] = 1000
    pi = np.zeros(N, dtype=int)
    pi[sources] = -1
    U = []
    
    Q = sorted(range(N), key=lambda v: b[v])
    for v in sinks:
        Q.remove(v)

    # run the Dijkstra algorithm
    while len(Q) > 0:
        w = Q.pop()
        U.append(w)
        
        # relax
        for v in G[1][np.where( G[0] == w )]:
            if b[v] < min(b[w], net_flux[w,v]):
                b[v] = min(b[w], net_flux[w,v])
                pi[v] = w

        Q = sorted(Q, key=lambda v: b[v])

    logger.info("Searched %s nodes", len(U)+len(sinks))

    return pi, b


def _back_relax(s, b, pi, NFlux):
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
            b, pi, NFlux = _back_relax(sprime, b, pi, NFlux)
            
    return b, pi, NFlux


def _backtrack(B, b, pi, NFlux):
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
                path.append( pi[path[-1]] )
        path.reverse()

        bottleneck, Flux = find_path_bottleneck(path, NFlux)

        logger.debug('In Backtrack: Flux %s, bestflux %s', Flux, bestflux)
        
        if Flux > bestflux: 
            bestpath = path
            bestflux = Flux

    if Flux == 0:
        bestpath = []
        bottleneck = (np.nan, np.nan)
        bestflux = 0

    return (bestpath, bottleneck, bestflux)


def find_path_bottleneck(path, net_flux):
    """
    Simply finds the bottleneck along a path. 
    
    This is the point at which the cost function first goes up along the path,
    backtracking from B to A.

    Parameters
    ----------
    path : list
        a list of nodes along the path of interest
    net_flux : matrix
        the net flux matrix

    Returns
    -------
    bottleneck : tuple
        a tuple of the nodes on either end of the bottleneck
    flux : float
        the flux at the bottleneck

    See Also
    --------
    find_top_paths : child function
        `find_top_paths` is probably the function you want to call to find
         paths through an MSM network. This is a utility function called by
         `find_top_paths`, but may be useful in some specific cases
    """

    if scipy.sparse.issparse(net_flux):
        net_flux = net_flux.tolil()
        
    flux = 100000. # initialize as large value

    for i in range(len(path) - 1):
        if net_flux[ path[i], path[i+1] ] < flux:
            flux = net_flux[ path[i], path[i+1] ]
            b1 = path[i]
            b2 = path[i+1]

    return (b1, b2), flux


def calculate_fluxes(sources, sinks, tprob, populations=None, committors=None):
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
    
    if scipy.sparse.issparse(tprob):
        dense = False
    else:
        dense = True
    
    # check if we got the populations
    if populations is None:
        eigens = msm_analysis.get_eigenvectors(tprob, 5)
        if np.count_nonzero(np.imag(eigens[1][:,0])) != 0:
            raise ValueError('First eigenvector has imaginary components')
        populations = np.real(eigens[1][:,0])
        
    # check if we got the committors
    if committors is None:
        committors = calculate_committors(sources, sinks, tprob)
    
    # perform the flux computation
    Indx, Indy = tprob.nonzero()

    n = tprob.shape[0]
    
    if dense:
        X = np.zeros((n, n))
        Y = np.zeros((n, n))
        X[(np.arange(n), np.arange(n))] = populations * (1.0 - committors)
        Y[(np.arange(n), np.arange(n))] = committors
    else:
        X = scipy.sparse.lil_matrix( (n,n) )
        Y = scipy.sparse.lil_matrix( (n,n) )
        X.setdiag( populations * (1.0 - committors) )
        Y.setdiag(committors)
    
    if dense:
        fluxes = np.dot( np.dot(X, tprob), Y )
        fluxes[( np.arange(n), np.arange(n) )] = np.zeros(n)
    else:
        fluxes = np.dot( np.dot(X.tocsr(), tprob.tocsr()), Y.tocsr() )
        fluxes = fluxes.tolil()
        fluxes.setdiag(np.zeros(n))
    
    return fluxes


def calculate_net_fluxes(sources, sinks, tprob, populations=None, committors=None):
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

    if scipy.sparse.issparse(tprob):
        dense = False
    else:
        dense = True

    n = tprob.shape[0]
    
    flux = calculate_fluxes(sources, sinks, tprob, populations, committors)
    ind = flux.nonzero()
    
    if dense:
        net_flux = np.zeros((n, n))
    else:
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

def calculate_ensemble_mfpt(sources, sinks, tprob, lag_time):
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

    X = calculate_mfpt(sinks, tprob, lag_time)
    times = np.zeros(len(sources))
    for i in range(len(sources)):
        times[i] = X[ sources[i] ] 

    return np.average(times), np.std(times) 


def calculate_avg_TP_time(sources, sinks, tprob, lag_time):
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

    n = tprob.shape[0]
    if scipy.sparse.issparse(tprob):
        T = tprob.tolil()
        P = scipy.sparse.lil_matrix((n, n))
    else:
        p = np.zeros((n, n))
    
    for u in sources:
        for i in range(n):
            if i not in sources:
                P[u,i] = T[u,i]

    for u in sources:
        T[u,:] = np.zeros(n)
        T[:,u] = 0

    for i in sources:
        N = T[i,:].sum()
        T[i,:] = T[i,:]/N

    X = calculate_mfpt(sinks, tprob, lag_time)
    TP = P * X.T
    TPtimes = []

    for time in TP:
        if time != 0: TPtimes.append(time)

    return np.average(TPtimes), np.std(TPtimes)


def calculate_mfpt(sinks, tprob, lag_time=1.):
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
    calculate_all_to_all_mfpt : function
        A more efficient way to calculate all the MFPTs in a network
    """
    
    sinks = _ensure_iterable(sinks)

    n = tprob.shape[0]
    
    if scipy.sparse.isspmatrix(tprob):
        tprob = tprob.tolil()
    
    for state in sinks:
        tprob[state,:] = 0.0    
        tprob[state,state] = 2.0

    if scipy.sparse.isspmatrix(tprob):
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


def calculate_all_to_all_mfpt(tprob, populations=None):
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
    
    if scipy.sparse.issparse(tprob):
        tprob = tprob.toarray()
        logger.warning('calculate_all_to_all_mfpt does not support sparse linear algebra')

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
    #z = scipy.linalg.inv(scipy.sparse.eye(num_states, num_states) - (tprob - limiting_matrix))
    z = scipy.linalg.inv(np.eye(num_states) - (tprob - limiting_matrix))

    # mfpt[i,j] = z[j,j] - z[i,j] / pi[j]
    mfpt = -z
    for j in range(num_states):
        mfpt[:, j] += z[j,j]
        mfpt[:, j] /= populations[j]

    return mfpt


def calculate_committors(sources, sinks, tprob):
    """
    Get the forward committors of the reaction sources -> sinks.
	
    Parameters
    ----------
    sources : array_like, int
        The set of unfolded/reactant states.
    sinks : array_like, int
        The set of folded/product states.		
    tprob : mm_matrix	
        The transition matrix.			
		
    Returns
    -------
    Q : array_like
        The forward committors for the reaction U -> F.
    """
	
    sources, sinks = _check_sources_sinks(sources, sinks)

    if scipy.sparse.issparse(tprob):
        dense = False
        tprob = tprob.tolil()
    else:
        dense = True

    # construct the committor problem
    n = tprob.shape[0]

    if dense:
       T = np.eye(n) - tprob
    else:
       T = scipy.sparse.eye(n, n, 0, format='lil') - tprob
       T = T.tolil()

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
    
    if dense:
        RHS = np.dot(tprob, IdB)
    else:
        RHS = tprob * IdB
    
    RHS[sources] = 0.0
    RHS[sinks]   = 1.0

    # solve for the committors
    if dense == False:
        Q = scipy.sparse.linalg.spsolve(T.tocsr(), RHS)
    else:
        Q = np.linalg.solve(T, RHS)
        
    assert np.all( Q <= 1.0 )
    assert np.all( Q >= 0.0 )

    return Q
    

######################################################################
# Functions for computing hub scores, conditional committors, and
# related quantities
#


def lump_transition_matrix(tprob, states_to_lump):
    """
    Lump a set of states in a transition matrix into a macrostate,
    combining the transition probabilities for that state.
    
    Parameters
    ----------
    tprob : matrix
        The original transition probability matrix to lump  
    states_to_lump: nd_array, int
        Indicies of the states to lump into a macrostate
    
    Returns
    -------
    lumped : matrix
        The lumped transition probability matrix.
    """
    
    # find out which indices we're keeping 
    new_N = tprob.shape[0] - len(states_to_lump) + 1
    to_keep_inds = range(tprob.shape[0])
    for i in states_to_lump:    # TJL: optimization point
        to_keep_inds.remove(i)
    m = len(to_keep_inds)
    
    # the new row is just the average transition probability
    new_row = np.sum( tprob[states_to_lump, :], axis=0 )[to_keep_inds] / \
                float(len(states_to_lump))
    assert len(new_row) == new_N - 1 
    
    # fill in the "lumped" matrix with the new data
    lumped = np.delete(tprob, states_to_lump, axis=0)
    lumped = np.delete(lumped, states_to_lump, axis=1)
    lumped = np.vstack([lumped, new_row])

    A = np.zeros((lumped.shape[0], 1))
    A[:,0] = 1.0 - np.sum(lumped, axis=1).flatten()
    lumped = np.hstack( [ lumped, A ] )
    
    assert lumped.shape[0] == lumped.shape[1]
    assert np.all( lumped.sum(1) == np.ones(lumped.shape[0]) )

    return lumped


def calculate_fraction_visits(tprob, waypoint, sources, sinks, 
                              Q=None, return_cond_Q=False):
    """
    Calculate the fraction of times a walker on `tprob` going from `sources` 
    to `sinks` will travel through the set of states `waypoints` en route.
    
    Computes the conditional committors q^{ABC^+} and uses them to find the
    fraction of paths mentioned above. The conditional committors can be 
    
    Note that in the notation of Dickson et. al.
        sources   = A
        sinks     = B
        waypoints = C
        
    Parameters
    ----------
    tprob : matrix
        The transition probability matrix        
    waypoint : int
        The index of the intermediate state
    sources : nd_array, int or int
        The indices of the source state(s)    
    sinks : nd_array, int or int
        The indices of the sink state(s)    
    return_cond_Q : bool
        Whether or not to return the conditional committors
    
    Returns
    -------
    fraction_paths : float
        The fraction of times a walker going from `sources` -> `sinks` stops
        by `waypoints` on its way.
    cond_Q : nd_array, float (optional)
        Optionally returned (`return_cond_Q`)
    
    Additional Parameters
    ---------------------
    Q : nd_array, float
        If you have the committors for the lumped transition matrix
        already calculated, pass them here to speed the computation.
        This argument is used heavily in the related function `hub_score`
        and is primarily intended for use there.
        
    See Also
    --------
    calculate_hub_score : function
        Compute the 'hub score', the weighted fraction of visits for an
        entire network.      
    calculate_all_hub_scores : function
        A more efficient way to compute the hub score for every state in a
        network.
        
    Notes
    -----
    Employs dense linear algebra,
      memory use scales as N^2
      cycle use scales as N^3      
      
    References
    ----------
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput., 
          Article ASAP DOI: 10.1021/ct300537s
    """
     
    # do some typechecking - we need to be sure that the lumped sources are in
    # the second to last row, and the lumped sinks are in the last row
    
    # check `tprob`
    if type(tprob) != np.ndarray:
        try:
            tprob = tprob.todense()
        except AttributeError as e:
            raise TypeError('Argument `tprob` must be convertable to a dense'
                            'numpy array. \n%s' % e)
    
    sources, sinks = _check_sources_sinks(sources, sinks)

    # rearrange the transition matrix so that row -2 are the lumped sources,
    # row -1 is the lumped sinks
    #tprob = lump_transition_matrix(tprob, sources)
    #tprob = lump_transition_matrix(tprob, sinks)
       
    # typecheck `waypoints` 
    if type(waypoint) == int:
        pass
    elif hasattr(waypoint, 'len')
        len(waypoint) == 1:
            waypoint = waypoint[0]
    else:
        raise TypeError('Argument `waypoint` must be an int')
    
    if np.any(sources == waypoint) or np.any(sinks == waypoint):
        raise ValueError('sources, sinks, waypoints must all be disjoint!')
        
    # if not provided, calculate the committors/lumped transition matrix
    if Q == None:
        Q = calculate_committors(sources, sinks, tprob)
        
    # construct absorbing Markov chain (T), remove all waypoints & lumped sink
    N = tprob.shape[0]
    #T = np.delete(tprob, waypoints + [N-1], axis=0)
    
    # permute the transition matrix into cannonical form - send waypoint the the
    # last row, and sinks to the end after that
    perm = np.arange(N)
    perm = np.delete(perm, sinks + [waypoint])
    perm = np.append(perm, sinks + [waypoint])
    print "perm", perm
    T = MSMLib.permute_tmat(tprob, perm)
    
    # extract P, R
    #n,m = T.shape
    n = N - len(sinks) - 1
    P = T[:n,:n]
    R = T[:n,n:]
    
    # calculate the conditional committors ( B = N*R )
    print P.sum(1)
    B = np.dot( np.linalg.inv( np.eye(n) - P ), R )
    assert B.shape == (n,m)
    
    # Now, B[i,-1] is the prob state i ends in a sink
    # while B[i,j] is the prob state i ends in waypoints[j]
    # we sum over all of the waypoints in the next line
    B_sum = np.zeros(N)
    B_sum[waypoints] = 1.0
    
    not_waypoint_not_sink = range(N)
    for i in waypoints + sinks:
        not_waypoint_not_sink.remove(i)

    B_sum[not_waypoint_not_sink] = B[:,:-1].sum(axis=1) # sum over "j"
    assert B_sum.shape == (N,)
    
    # we need to "add back in" the values for the waypoints - note the committor 
    # conditional on C for a state in C is just the original committor
    cond_Q = Q * B_sum
    print "cond_Q", type(cond_Q), cond_Q.shape, (N, N)
    #assert cond_Q.shape == (N, N)
    print cond_Q
    assert np.all( cond_Q <= 1.0 )
    assert np.all( cond_Q >= 0.0 )
    
    # finally, calculate the fraction of paths h_C(A,B)
    fraction_paths = np.sum( cond_Q[:-2,:-2] * tprob[:-2,:-2]) / \
                        ( tprob[-2,-1] + np.sum( Q[:-2,:-2] * tprob[:-2,:-2] ) )
    
    if return_cond_Q:
        return fraction_paths, cond_Q
    else:
        return fraction_paths


def calculate_hub_score(tprob, waypoints):
    """
    Calculate the hub score for the set of states `waypoints` - which can
    either be a list/array of microstates (i.e. a macrostate) or a single
    microstate.
    
    The "hub score" is a measure of how well traveled a certain state or
    set of states is in a network. Specifically, it is the fraction of
    times that a walker visits a state en route from some state A to another
    state B, averaged over all combinations of A and B.
    
    
    Parameters
    ----------
    tprob : matrix
        The transition probability matrix        
    waypoints : nd_array, int or int
        The indices of the intermediate state(s)
                
    Returns
    -------
    Hc : float
        The hub score for the state composed of `waypoints`

    See Also
    --------
    calculate_fraction_visits : function
        Calculate the fraction of times a state is visited on pathways going
        from a set of "sources" to a set of "sinks".        
    calculate_all_hub_scores : function
        A more efficient way to compute the hub score for every state in a
        network.
        
    Notes
    -----
    Employs dense linear algebra,
      memory use scales as N^2
      cycle use scales as N^5

    References
    ----------
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput., 
        Article ASAP DOI: 10.1021/ct300537s
    """
    
    # typecheck
    if (type(waypoints) or type(waypoints)) == list:
        pass
    elif type(waypoints) == int:
        waypoints = np.array([waypoints])
    else:
        raise ValueError('Must pass waypoints as int or list/array of ints')
    
    # find out which states to include in A, B (i.e. everything but C)
    N = tprob.shape[0]
    states_to_include = range(N) # TJL: optimization point
    for w in waypoints:
        states_to_include.remove(w)
    
    # calculate the hub score
    Hc = 0.0
    for s1 in states_to_include:
        for s2 in states_to_include:
            if (s1 != s2) and (s1 not in waypoints) and (s2 not in waypoints):
                Hc += calculate_fraction_visits(tprob, waypoints, s1, s2, 
                                                Q=None, return_cond_Q=False)
    
    Hc /= ( (N-1) * (N-2) )
                
    return Hc
    
    
def calculate_all_hub_scores(tprob):
    """
    Calculate the hub scores for all states in a network defined by `tprob`.
    
    The "hub score" is a measure of how well traveled a certain state or
    set of states is in a network. Specifically, it is the fraction of
    times that a walker visits a state en route from some state A to another
    state B, averaged over all combinations of A and B.
        
    Parameters
    ----------
    tprob : matrix
        The transition probability matrix        
        
    Returns
    -------
    Hc_array : nd_array, float
        The hub score for each state in `tprob`

    See Also
    --------
    calculate_fraction_visits : function
        Calculate the fraction of times a state is visited on pathways going
        from a set of "sources" to a set of "sinks".        
    calculate_hub_score : function
        A function that computes just one hub score, can compute the hub score
        for a set of states.
                
    Notes
    -----
    Employs dense linear algebra,
      memory use scales as N^2
      cycle use scales as N^6

    References
    ----------
    ..[1] Dickson & Brooks (2012), J. Chem. Theory Comput., 
        Article ASAP DOI: 10.1021/ct300537s
    """
    
    N = tprob.shape[0]
    states = range(N)
    
    # calculate the hub score
    Hc_array = np.zeros(N)
    
    # loop over each state and compute it's hub score
    for i,waypoint in enumerate(states):
        
        Hc = 0.0
        Q = calculate_all_to_all_mfpt(tprob) # we can save time by re-using the committors
        
        # now loop over all combinations of sources/sinks and average
        for s1 in states:
            if waypoint != s1:
                for s2 in states:
                    if s1 != s2:
                        if waypoint != s2:
                            Hc += calculate_fraction_visits(tprob, waypoint, s1, s2, Q)
        
        # store the hub score in an array
        Hc_array[i] = Hc / ( (N-1) * (N-2) )
    
    return Hc_array


######################################################################
#  ALIASES FOR DEPRECATED FUNCTION NAMES
#  THESE FUNCTIONS WERE ADDED FOR VERSION 2.6 AND
#  CAN BE REMOVED IN VERSION 2.7
######################################################################

@deprecated(calculate_committors, '2.7')
def GetBCommittorsEqn(U, F, T0, EquilibriumPopulations):
    pass
    
@deprecated(calculate_committors, '2.7')
def GetFCommittorsEqn(A, B, T0, dense=False):
    pass

# TJL: We don't need a "get backwards committors", since they are just
# 1 minus the forward committors
@deprecated(calculate_committors, '2.7')
def GetBCommittors(U, F, T0, EquilibriumPopulations, X0=None, Dense=False):
    pass

@deprecated(calculate_mfpt, '2.7')
def MFPT():
    pass

@deprecated(calculate_committors, '2.7')
def GetFCommittors():
    pass
    
@deprecated(calculate_fluxes, '2.7')
def GetFlux():
    print "WARNING: The call signature for the new function has changed."
    pass

@deprecated(calculate_net_fluxes, '2.7')
def GetNetFlux():
    print "WARNING: The call signature for the new function has changed."
    pass
    
@deprecated(calculate_avg_TP_time, '2.7')
def CalcAvgTPTime():
    pass
    
@deprecated(calculate_ensemble_mfpt, '2.7')
def CalcAvgFoldingTime():
    pass
    
