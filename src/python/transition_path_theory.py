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

"""Functions for calculating Transition Path Theory calculations.

Examples:
Mean First Passage Times
Transition Path Theory Fluxes
Pathways
Committors
"""

debug = False # set to increase verbosity for coding purposes

import sys
import numpy as np
import scipy.sparse
from msmbuilder import MSMLib


def DijkstraTopPaths(A, B, NFlux, NumPaths=10, NodeWipe=False):
    r"""
    Calls the Dijkstra algorithm to find the top 'NumPaths'. 

    Does this recursively by first finding the top flux path, then cutting that
    path and relaxing to find the second top path. Continues until NumPaths
    have been found.

    Parameters
    ----------
    A : array_like, int 
        The indices of the source states (i.e. for state A in rxn A -> B)
    B : array_like, int
        Indices of sink states (state B)
    NFlux : sparse matrix
        Matrix of the net flux from A to B, see function GetFlux
    NumPaths : int 
        The number of paths to find
    NodeWipe : bool 
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
    """


    # first, do some checking on the input, esp. A and B
    # we want to make sure all objects are iterable and the sets are disjoint
    try:
        l = len(A)
    except:
        A = list([int(A)])
        print "Warning: passed object 'A' was not iterable, converted it to:", A
    try: 
        l = len(B)
    except: 
        B = list([int(B)])
        print "Warning: passed object 'B' was not iterable, converted it to:", B
    if np.any( A == B ):
        raise ValueError("Sets A and B must be disjoint to find paths between them")

    # initialize objects
    Paths = []
    Fluxes = []
    Bottlenecks = []
    NFlux = NFlux.tolil()

    # run the initial Dijkstra pass
    pi, b = Dijkstra(A, B, NFlux)

    print "Path Num | Path | Bottleneck | Flux" 

    i = 1
    done = False
    while not done:

        # First find the highest flux pathway
        (Path, (b1,b2), Flux) = Backtrack(B, b, pi, NFlux)

        # Add each result to a Paths, Bottlenecks, Fluxes list
        if Flux == 0:
            print "Only %d possible pathways found. Stopping backtrack." % i
            break
        Paths.append(Path)
        Bottlenecks.append( (b1,b2) )
        Fluxes.append(Flux)
        print i, Path, (b1, b2), Flux

        # Cut the bottleneck, start relaxing from B side of the cut
        if NodeWipe: 
            NFlux[:, b2] = 0
            print "Wiped node:", b2
        else: NFlux[b1, b2] = 0

        G = scipy.sparse.find(NFlux)
        Q = [b2]
        b, pi, NFlux = BackRelax(b2, b, pi, NFlux)
        
        # Then relax the graph and repeat
        # But only if we still need to
        if i != NumPaths-1:
            while len(Q) > 0:
                w = Q.pop()
                for v in G[1][np.where( G[0] == w )]:
                    if pi[v] == w:
                        b, pi, NFlux = BackRelax(v, b, pi, NFlux)
                        Q.append(v)
                Q = sorted(Q, key=lambda v: b[v])

        i+=1
        if i == NumPaths+1: 
            done = True
        if Flux == 0: 
            print "Only %d possible pathways found. Stopping backtrack." % i
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

    print "Searched", len(U)+len(B), "nodes"

    return pi, b


def BackRelax(s, b, pi, NFlux):
    """
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
        if debug: print 'In Backtrack: Flux, bestflux:', Flux, bestflux
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


def CalcAvgFoldingTime(U, F, T, LagTime):
    """
    Calculates the Average 'Folding Time' of an MSM defined by T and a LagTime.
    The Folding Time is the average of the MFPTs (to F) of all the states in U.

    Note here 'Folding Time' is defined as the avg MFPT of {U}, to {F}.
    Consider this carefully. This is probably NOT the experimental folding time!

    Parameters
    ----------
    U : array, int
        indices of the unfolded states
    F : array, int 
        indices of the folded states
    T : matrix
        transition probability matrix
    LagTime : float
        the lag time used to create T (dictates units of the answer)

    Returns
    -------
    avg : float
        the average of the MFPTs
    std : float
        the standard deviation of the MFPTs
    """

    X=GetMFPT(F,T,LagTime)
    Times=np.zeros(len(U))
    for i in range(len(U)):
        Times[i]=(X[U[i]])

    return np.average(Times), np.std(Times) 


def CalcAvgTPTime(U, F, T, LagTime):
    """
    Calculates the Average Transition Path Time for MSM with: T, LagTime.
    The TPTime is the average of the MFPTs (to F) of all the states
    immediately adjacent to U, with the U states effectively deleted.

    Note here 'TP Time' is defined as the avg MFPT of all adjacent states to {U},
    to {F}, ignoring {U}.

    Consider this carefully.

    Parameters
    ----------
    U : array, int
        indices of the unfolded states
    F : array, int 
        indices of the folded states
    T : matrix
        transition probability matrix
    LagTime : float
        the lag time used to create T (dictates units of the answer)

    Returns
    -------
    avg : float
        the average of the MFPTs
    std : float
        the standard deviation of the MFPTs
    """

    T=T.tolil()
    n=T.shape[0]
    P=scipy.sparse.lil_matrix((n,n))

    for u in U:
        for i in range(n):
            if i not in U:
                P[u,i]=T[u,i]

    for u in U:
        T[u,:]=np.zeros(n)
        T[:,u]=0

    for i in U:
        N=T[i,:].sum()
        T[i,:]=T[i,:]/N

    X=GetMFPT(F,T,LagTime)
    TP=P*X.T
    TPtimes=[]

    for time in TP:
        if time!=0: TPtimes.append(time)

    return np.average(TPtimes), np.std(TPtimes)


def GetMFPT(F, T, LagTime=1., tolerance=10**-10, maxiter=50000):
    """
    Gets the Mean First Passage Time (MFPT) for all states to a *set*
    of sinks.

    Parameters
    ----------
    F : array, int 
        indices of the folded states
    T : matrix
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

    # It appears that the `tolerance` and `maxiter` arguments are not being used...?
    # can someone confirm they are deprecated? (Kyle?!?) --TJL

    n=T.shape[0]
    T2=T.copy().tolil()
    for State in F:
        T2[State,:]=0.0     # CRS
        T2[State,State]=2.0 # CRS
        #T2[:,State]=0.0     # CRS

    T2=T2-scipy.sparse.eye(n,n)
    T2=T2.tocsr()

    RHS=-1*np.ones(n)
    for State in F:
        RHS[State]=0.

    MFPT=LagTime*scipy.sparse.linalg.spsolve(T2,RHS)

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
        eigens = MSMLib.GetEigenvectors(tprob,5)
        if np.count_nonzero(np.imag(eigens[1][:,0])) != 0:
            raise ValueError('First eigenvector has imaginary parts')
        populations = np.real(eigens[1][:,0])

    # ensure that tprob is a transition matrix
    MSMLib.CheckTransition(tprob)
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


def GetBCommittors(U,F,T0,EquilibriumPopulations,maxiter=100000,X0=None,Dense=False):
    """Get the backward committors of the reaction U -> F.  T0 is the transition matrix.  EquilibriumPopulations are required for the backward committors but not the foward commitors because we assume detailed balance when calculating the backward committors.  If you are have small matrices, it can be faster to use dense linear algebra.  If you don't use dense linear algebra, you might want to specify an initial vector X0 and the maximum number of iterations for controlling the sparse equation solver."""
    A,b,Q0=GetBCommittorsEqn(U,F,T0,EquilibriumPopulations)
    if X0!=None:
        Q0=X0

    if Dense==False:
        Q=SolveEqn(A,b,Q0,maxiter)
    else:
        Q=np.linalg.solve(A.toarray(),b)
    return Q


def GetFCommittors(U,F,T0,maxiter=100000,X0=None,Dense=False):
    """Get the forward committors of the reaction U -> F.  T0 is the transition matrix.  If you are have small matrices, it can be faster to use dense linear algebra.  If you don't use dense linear algebra, you might want to specify an initial vector X0 and the maximum number of iterations for controlling the sparse equation solver."""
    A,b,Q0=GetFCommittorsEqn(U,F,T0)
    if X0!=None:
        Q0=X0
    if Dense==False:
        Q=SolveEqn(A,b,Q0,maxiter)
    else:
        Q=np.linalg.solve(A.toarray(),b)
    return(Q)


def GetBCommittorsEqn(U,F,T0,EquilibriumPopulations):
    """Construct the matrix equations used for finding backwards committors for the reaction U -> F.  T0 is the transition matrix, Equilibruim is the vector of equilibruim populations"""

    n=len(EquilibriumPopulations)
    DE=scipy.sparse.eye(n,n,0,format='lil')
    DE.setdiag(EquilibriumPopulations)
    DEInv=scipy.sparse.eye(n,n,0,format='lil')
    DEInv.setdiag(1./EquilibriumPopulations)
    TR=(DEInv.dot(T0.transpose())).dot(DE)

    return(GetFCommittorsEqn(F,U,TR))


def GetFCommittorsEqn(A,B,T0,maxiter=100000):
    """Construct the matrix equations used for finding committors for the reaction U -> F.  T0 is the transition matrix, Equilibruim is the vector of equilibruim populations"""
    n=T0.shape[0]
    T=scipy.sparse.eye(n,n,0,format='lil')-T0
    T=T.tolil()
    for a in A:
        T[a,:]=np.zeros(n)
        T[:,a]=0.0
        T[a,a]=1.0
    for b in B:
        T[b,:]=np.zeros(n)
        T[:,b]=0.0
        T[b,b]=1.0
    IdB=np.zeros(n)
    for b in B:
        IdB[b]=1.0
    print "done with setting up matrices"
    RHS=T0*(IdB) # changed from RHS=T0.matvec(IdB) --TJL, matvec deprecated
    for a in A:
        RHS[a]=0.0
    for b in B:
        RHS[b]=1.0
    Q0=np.ones(n)
    for a in A:
        Q0[a]=0.0
    return(T,RHS,Q0)


def GetFlux(E,F,R,T):
    """Get the flux matrix.
    INPUT
    E - the equilibirum populations
    F - forward committors
    R - backwards committors
    T - transition matrix. """
    Indx,Indy=T.nonzero()

    n=len(E)
    X=scipy.sparse.lil_matrix((n,n))
    X.setdiag(E*R)

    Y=scipy.sparse.lil_matrix((n,n))
    Y.setdiag(F)
    P=np.dot(np.dot(X.tocsr(),T.tocsr()),Y.tocsr())
    P=P.tolil()
    P.setdiag(np.zeros(n))
    return(P)


def GetNetFlux(E,F,R,T):
    """Returns a (net) matrix Flux where Flux[i,j] is the flux from i to j.

    INPUT
    E - the equilibirum populations
    F - forward committors
    R - backwards committors
    T - transition matrix. 
    """

    n=len(E)
    Flux=GetFlux(E,F,R,T)
    ind=Flux.nonzero()
    NFlux=scipy.sparse.lil_matrix((n,n))
    for k in range(len(ind[0])):
        i,j=ind[0][k],ind[1][k]
        forward=Flux[i,j]
        reverse=Flux[j,i]
        NFlux[i,j]=max(0,forward-reverse)
    return(NFlux)
        
    
def AddPathwaysToMatrix(F, Pathway, PathFlux, Verbose=False):
    """Given a matrix F that is to contain a subset of the fluxes, add Pathway with flux PathFlux."""
    for k in range(len(Pathway)):
        if Verbose: print(k)
        P=Pathway[k]
        Flux=PathFlux[k]
        for i in range(len(P)-1):
            F[P[i],P[i+1]]+=Flux

            
def SolveEqn(A, b, X, nIter):
    """This is a simple Jacobi solver for the equation Ax=b.  X is the starting vector, nIter is the number of iterations."""
    n=A.shape[0]
    #D=scipy.sparse.lil_diags([A.diagonal()],[0],(n,n))
    D=scipy.sparse.eye(n,n,0,format='lil')
    D.setdiag(A.diagonal())
    R=A-D
    #DI=scipy.sparse.lil_diags([1/A.diagonal()],[0],(n,n))
    DI=scipy.sparse.eye(n,n,0,format='lil')
    DI.setdiag(1./A.diagonal())
    DI=DI.tocsr()
    R=R.tocsr()
    for i in range(nIter):
        X=DI*((b-R*X)) #changed from X=DI.matvec((b-R.matvec(X)))
    return(X)
    
