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

import numpy as np
import scipy.sparse

def DijkstraTopPaths(A, B, NFlux, NumPaths=10, NodeWipe=False):
    """
    Calls the Dijkstra algorithm to find the top 'NumPaths'. 
    Does this recursively by first finding the top flux path, then cutting that
    path and relaxing to find the second top path. Continues until NumPaths
    have been found.

    Input:
    A - array of ints for state A in rxn A -> B
    B - array of ints for state B
    NFlux - sparse flux matrix
    NumPaths - int representing the number of paths to find
    NodeWipe - removes the bottleneck-generating node from the graph, instead of just the bottleneck.
    Not generally recommended, but good for qualitative elucidation of heterogeneous paths.

    Returns:
    Paths - a list of lists of the nodes transversed in each path
    Bottlenecks - a list of tuples of nodes, between which exists the path bottleneck
    Fluxes - a list of floats, representing the flux through each path
    RETURNED FORMAT: (Paths, Bottlenecks, Fluxes)
    """

    Paths=[]
    Fluxes=[]
    Bottlenecks=[]
    NFlux = NFlux.tolil()

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
        if i == NumPaths: done = True
        if Flux == 0: 
            print "Only %d possible pathways found. Stopping backtrack." % i
            done = True

    return Paths, Bottlenecks, Fluxes


def Dijkstra(A, B, NFlux):
    """
    A modified Dijkstra algorithm that dynamically computes the cost
    of all paths from A to B, weighted by NFlux.

    Input:
    A - array of ints for state A in rxn A -> B
    B - array of ints for state B
    NFlux - sparse flux matrix

    Returns:
    pi - an array representing the paths from A->B, pi[i] = node preceeding i
    b - the flux passing through each node
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
    Cuts relaxes the B-side of a cut edge (node b2) to source from the
    adjacent node with the most flux flowing to it. If there are no
    adjacent source nodes, cuts the node out of the graph and relaxes the
    nodes that were getting fed by b2.

    s - the node b2
    b - the cost function
    pi - the backtrack array
    NFlux - net flux matrix
    """
    G = scipy.sparse.find(NFlux)
    if len( G[0][np.where( G[1] == s )] ) > 0:
        # Resource that node from the best option one level lower
        # Notation: j is node one level below, s is the one being considered
        b[s] = 0
        for j in G[0][np.where( G[1] == s )]:
            if b[s] < min( b[j], NFlux[j,s]):
                b[s] = min( b[j], NFlux[j,s])
                pi[s] = j
    else: 
        for sprime in G[1][np.where( G[0] == s )]:
            NFlux[s,sprime]=0
            b, pi, NFlux = BackRelax(sprime, b, pi, NFlux)
            
    return b, pi, NFlux


def Backtrack(B, b, pi, NFlux):
    """
    Works backwards to pull out a path from pi, where pi is a list such that
    pi[i] = source node of node i. Begins at the largest staring incoming flux
    point in B.
    """

    # Select starting location
    bestflux = 0
    for Bnode in B:
        path = [Bnode]
        NotDone=True
        while NotDone:
            if pi[path[-1]] == -1: break
            else: path.append(pi[path[-1]])
        path.reverse()

        ( (b1, b2), Flux) = FindPathBottleneck(path, NFlux)
        if Flux > bestflux: 
            bestpath = path
            bestflux = Flux
            bottleneck = (b1,b2)

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
    """
    NFlux=NFlux.tolil()
    MinVal = 100
    for i in range(len(Path)-1):
        if NFlux[ Path[i], Path[i+1] ] < MinVal:
            MinVal = NFlux[ Path[i], Path[i+1] ]
            b1 = Path[i]
            b2 = Path[i+1]
    return ( (b1, b2), MinVal)


def CalcAvgFoldingTime(U,F,T,LagTime):
    """
    Calculates the Average 'Folding Time' of an MSM defined by T and a LagTime.
    The Folding Time is the average of the MFPTs (to F) of all the states in U.

    Note here 'Folding Time' is defined as the avg MFPT of {U}, to {F}.
    Consider this carefully.

    Arguments:
    - U: 1D array of unfolded states
    - F: 1D array of folded states
    - T: transition probability matrix
    - LagTime: the lag time used to create T

    Returns:
    - two floats: The avg folding time and its STD
    """
    X=GetMFPT(F,T,LagTime)
    Times=np.zeros(len(U))
    for i in range(len(U)):
        Times[i]=(X[U[i]])
    return (np.average(Times),np.std(Times))


def CalcAvgTPTime(U,F,T,LagTime):
    """
    Calculates the Average Transition Path Time for MSM with: T, LagTime.
    The TPTime is the average of the MFPTs (to F) of all the states
    immediately adjacent to U, with the U states effectively deleted.

    Note here 'TP Time' is defined as the avg MFPT of all adjacent states to {U},
    to {F}, ignoring {U}.
    Consider this carefully.

    Arguments:
    - U: 1D array of unfolded states
    - F: 1D array of folded states
    - T: transition probability matrix
    - LagTime: the lag time used to create T

    Returns:
    - a float: The avg TP time and its STD
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
    return(np.average(TPtimes),np.std(TPtimes))


def GetMFPT(F,T,LagTime=1.):
    """
    Gets the Mean First Passage Time all states in T

    Arguments:
    - F: 1D array of folded states
    - T: transition probability matrix
    - LagTime: The lag time used to construct T
    
    Returns:
    - X: an array of floats, MFPT to F in time units of LagTime, for each state
    """

    n=T.shape[0]
    T2=T.copy().tolil()
    for State in F:
        T2[State,:]=0.0# CRS
        T2[State,State]=2.0# CRS

    T2=T2-scipy.sparse.eye(n,n)
    T2=T2.tocsr()

    RHS=-1*np.ones(n)
    for State in F:
        RHS[State]=0.

    MFPT=LagTime*scipy.sparse.linalg.spsolve(T2,RHS)
    return(MFPT)



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
    #DE=scipy.sparse.lil_diags([EquilibriumPopulations],[0],(n,n))
    DE=scipy.sparse.eye(n,n,0,format='lil')
    DE.setdiag(EquilibriumPopulations)
    #DEInv=scipy.sparse.lil_diags([1/EquilibriumPopulations],[0],(n,n))
    DEInv=scipy.sparse.eye(n,n,0,format='lil')
    DEInv.setdiag(1./EquilibriumPopulations)
    TR=(DEInv.dot(T0.transpose())).dot(DE)
    return(GetFCommittorsEqn(F,U,TR))
def GetFCommittorsEqn(A,B,T0,maxiter=100000):
    """Construct the matrix equations used for finding committors for the reaction U -> F.  T0 is the transition matrix, Equilibruim is the vector of equilibruim populations"""
    n=T0.shape[0]
    #T=scipy.sparse.lil_eye((n,n))-T0#scipy.sparse.lil_eye deprecated
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
    print ("done with setting up matrices")
    RHS=T0*(IdB) # changed from RHS=T0.matvec(IdB)
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
        
def GreedyBacktrack(A,b,F):
    """Starting at the final state b, work backwards towards the initial set A while trying to find the maximal flux path along the flux matrix F."""
    Finished=False
    ind=b
    Pathway=[]
    Fluxes=[]
    while not Finished:
        print(ind)
        Data=F[:,ind].toarray().flatten()
        ind=np.argmax(Data)
        Pathway.append(ind)
        Fluxes.append(Data[ind])
        if ind in A:
            Finished=True
    PathFlux=Fluxes[np.argsort(Fluxes)[0]]
    Pathway.reverse()
    Pathway.append(b)
    print(Fluxes)
    print(PathFlux)
    for i in range(len(Pathway)-1):
        print(Pathway[i],Pathway[i+1])
        F[Pathway[i],Pathway[i+1]]-=PathFlux
    return(Pathway,PathFlux)

def GreedyBacktrackWithPFold(A,b,F,PFold, Verbose=False):
    """Starting at the final state b, work backwards towards the initial set A while trying to find the maximal flux path along the flux matrix F.  This function  enforces the constraint that the PFold value is nonincreasing along the path."""
    Finished=False
    ind=b
    Pathway=[]
    Fluxes=[]
    CurPFold=1.
    while not Finished:
        if Verbose: print(ind)
        Data=F[:,ind].toarray().flatten()
        PFoldInd=np.where(PFold<CurPFold)
        ind=np.argmax(Data[PFoldInd])
        ind=PFoldInd[0][ind]
        if Verbose: print(ind,Data[ind])
        Pathway.append(ind)
        Fluxes.append(Data[ind])
        CurPFold=PFold[ind]
        if ind in A:
            Finished=True
    PathFlux=Fluxes[np.argsort(Fluxes)[0]]
    if Verbose: print(PathFlux)
    Pathway.reverse()
    Pathway.append(b)
    if Verbose: print(Fluxes)
    if Verbose: print(PathFlux)
    for i in range(len(Pathway)-1):
        if Verbose: print(Pathway[i],Pathway[i+1])
        F[Pathway[i],Pathway[i+1]]-=PathFlux
    return(Pathway,PathFlux)

def GreedyBacktrackWithPFoldStochastic(A,b,F,PFold, beta=4.0, Verbose=False):
    """A Stochastic version of GreedyBacktrackWithPFold():
    Instead of always taking the largest-flux path, paths are chosen 
    stochastically according to the normalized distribution P(flux)^beta.

        beta = an inverse temperature (beta must be > 0)

    When beta > 1, the max-flux paths are more heavily weighted, with max-flux
    being the limiting case of beta -> inf.

    When beta < 1, the search is weighted across a more uniform distribution of
    fluxes, with a random search in the limit of beta -> 0

    The default of beta=4.0 worked well in iniial tests
          
    """

    Finished=False
    ind=b
    Pathway=[]
    Fluxes=[]
    CurPFold=1.
    while not Finished:
        if Verbose: print(ind)
        Data=F[:,ind].toarray().flatten()
        PFoldInd=np.where(PFold<CurPFold)
        # find the probability of taking each path
        DataProb = Data[PFoldInd]**beta
        DataProb = DataProb/DataProb.sum() # normalize
        r = np.random.rand()
        cumprob = 0.0
        for i in range(DataProb.shape[0]): 
            cumprob += DataProb[i]
            if cumprob > r:
                ind = i
                break 
        #ind=np.argmax(Data[PFoldInd])
        ind=PFoldInd[0][ind]
        if Verbose: print(ind,Data[ind])
        Pathway.append(ind)
        Fluxes.append(Data[ind])
        CurPFold=PFold[ind]
        if ind in A:
            Finished=True
    PathFlux=Fluxes[np.argsort(Fluxes)[0]]
    if Verbose: print(PathFlux)
    Pathway.reverse()
    Pathway.append(b)
    if Verbose: print(Fluxes)
    if Verbose: print(PathFlux)
    for i in range(len(Pathway)-1):
        if Verbose: print(Pathway[i],Pathway[i+1])
        F[Pathway[i],Pathway[i+1]]-=PathFlux
    return(Pathway,PathFlux)


def GreedyBacktrackWithPFoldStochasticScrub(A,b,F,PFold, beta=4.0, NScrubs=100, Verbose=False):
    """Like the stochastic version of GreedyBacktrackWithPFoldStochastic(),
    but each next top path is determined by a stochastic "scrub" to try to find the ACTUAL 
    top flux at each step.
    """

    BestPathway = []
    BestPathFlux = 0.0

    for scrub in range(NScrubs):

      Finished=False
      ind=b
      Pathway=[]
      Fluxes=[]
      CurPFold=1.
      while not Finished:
        if Verbose: print(ind)
        Data=F[:,ind].toarray().flatten()
        PFoldInd=np.where(PFold<CurPFold)
        # find the probability of taking each path
        DataProb = Data[PFoldInd]**beta
        DataProb = DataProb/DataProb.sum() # normalize
        r = np.random.rand()
        cumprob = 0.0
        for i in range(DataProb.shape[0]):
            cumprob += DataProb[i]
            if cumprob > r:
                ind = i
                break
        #ind=np.argmax(Data[PFoldInd])
        ind=PFoldInd[0][ind]
        if Verbose: print(ind,Data[ind])
        Pathway.append(ind)
        Fluxes.append(Data[ind])
        CurPFold=PFold[ind]
        if ind in A:
            Finished=True
      PathFlux=Fluxes[np.argsort(Fluxes)[0]]
      if Verbose: print(PathFlux)
      Pathway.reverse()
      Pathway.append(b)
      if Verbose: print(Fluxes)
      if Verbose: print(PathFlux)

      if PathFlux > BestPathFlux:
          BestPathway = Pathway
          BestPathFlux = PathFlux
          if Verbose: print 'Found better path on scrub', scrub, Pathway, BestPathFlux

    for i in range(len(BestPathway)-1):
      if Verbose: print(BestPathway[i],BestPathway[i+1])
      F[BestPathway[i],BestPathway[i+1]]-=BestPathFlux

    return(BestPathway,BestPathFlux)

    
def AddPathwaysToMatrix(F,Pathway,PathFlux, Verbose=False):
    """Given a matrix F that is to contain a subset of the fluxes, add Pathway with flux PathFlux."""
    for k in range(len(Pathway)):
        if Verbose: print(k)
        P=Pathway[k]
        Flux=PathFlux[k]
        for i in range(len(P)-1):
            F[P[i],P[i+1]]+=Flux
            
def SolveEqn(A,b,X,nIter):
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
    Xlast = np.ones( X.shape ) * 10000
    for i in range(nIter):
        X=DI*((b-R*X)) #changed from X=DI.matvec((b-R.matvec(X)))
        #err = abs(X-Xlast).max()
        print i#,err
        #if err <= 1E-10:
        #    break
        #Xlast = X.copy()
    return(X)
    
