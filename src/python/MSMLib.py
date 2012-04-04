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

"""MSMLib contains classes and functions for working with Transition and Count Matrices.

Notes:
1.  Assignments typically refer to a numpy array of integers such that Assignments[i,j] gives the state of trajectory i, frame j.
2.  Transition and Count matrices are typically stored in scipy.sparse.csr_matrix format.

MSMLib functions generally relate to one of the following:
1.  Counting the number of transitions observed in Assignment data--e.g., constructing a Count matrix.
2.  Constructing a transition matrix from a count matrix.
3.  Performing calculations with Assignments, Counts matrices, or transition matrices.
"""

import scipy.sparse
import scipy.linalg
import scipy
import numpy as np
import multiprocessing
import sys
import scipy.optimize
from collections import defaultdict

from Emsmbuilder import Serializer

eig=scipy.linalg.eig
sys.setrecursionlimit(200000)#The recursion limit is necessary for using the Tarjan algorithm, which is recursive and often requires quite a few iterations.
DisableErrorChecking=False#Set this value to true (MSMLib.DisableErrorChecking=True) to ignore Eigenvector calculation errors.  Useful if you need to process disconnected data.
MinimumAllowedNumEig=5

# try to import scipy sparse methods correctly, accounting for different namespaces in different version
def importSparseEig():
    try:
        import scipy.sparse.linalg
        sparseEigen = scipy.sparse.linalg.eigs
    except:
        pass
    else:
        return sparseEigen

    try:
        import scipy.sparse.linalg.eigen.arpack as arpack
        sparseEigen = arpack.eigen
    except:
        pass
    else:
        return sparseEigen

    try:
        import scipy.sparse.linalg.eigen
        sparseEigen = scipy.sparse.linalg.eigen.eigs
    except:
        pass
        try:
                import scipy.sparse.linalg.eigen
                sparseEigen = scipy.sparse.linalg.eigen
        except:
                pass
    else:
        return sparseEigen

    raise ImportError
sparseEigen = importSparseEig()

def flatten(*args):
    """Return a generator for a flattened form of all arguments"""

    for x in args:
        if hasattr(x, '__iter__'):
            for y in flatten(*x):
                yield y
        else:
            yield x

def IsTransitionMatrix(T,Epsilon=.00001):
    """Return true if T is a row normalized transition matrix, false otherwise."""

    n=T.shape[0]
    X=np.array(T.sum(1)).flatten()
    if scipy.linalg.norm(X-np.ones(n)) < Epsilon:
        return True
    else:
        return False

NormalizationError=Exception("Not a Row Normalized Matrix","Not a Row Normalized Matix")

def AreAllDimensionsSame(*args):
    """Find the shape of every input.  Return True if every matrix and vector is the same size."""

    m=len(args)
    DimList=[]
    for i in range(m):
        Dims=scipy.shape(args[i])
        DimList.append(Dims)
    FlatDimList=[x for x in flatten(DimList)]
    if min(FlatDimList)!=max(FlatDimList):
        return(False)
    else:
        return(True)

DimensionError=Exception("Argument has incorrect shape", "Argument has incorrect shape")

def CheckDimensions(*args):
    """Throw exception if one of the arguments has the incorrect shape."""

    if AreAllDimensionsSame(*args)==False:
        raise DimensionError
            
def CheckTransition(T,Epsilon=.00001):
    """Throw an exception if T is not a row normalized stochastic matrix."""
    if DisableErrorChecking:
        return
    if IsTransitionMatrix(T,Epsilon=Epsilon)==False:
        print(T)
        print("T is not a row normalized stocastic matrix.  This is often caused by either numerical inaccuracies or by having states with zero counts.")
        raise NormalizationError    

def GetTransitionCountMatrixSparse(states, numstates = None, LagTime = 1, slidingwindow = True):
    """Computes the transition count matrix for a sequence of states.

    Inputs:
    ----------
    states : array
    A one-dimensional array of integers representing the sequence of states.
    These integers must be in the range [0, numstates]
    
    numstates : integer
    The total number of states. If not specified, the largest integer in the
    states array plus one will be used.
    
    LagTime : integer
    The time delay over which transitions are counted
    
    slidingwindow : bool
    
    Returns
    -------
    C : sparse matrix of integers
    The computed transition count matrix
    """
  
    if not numstates:
        numstates = np.max(states)+1

    if slidingwindow:
        from_states = states[:-LagTime:1]
        to_states = states[LagTime::1]
    else:
        from_states = states[:-LagTime:LagTime]
        to_states = states[LagTime::LagTime]
    assert from_states.shape == to_states.shape

    transitions = np.row_stack((from_states,to_states))
    counts = np.ones(transitions.shape[1], dtype=int)
    try:
        C = scipy.sparse.coo_matrix((counts, transitions),shape=(numstates,numstates))
    except ValueError:
        # Lutz: if we arrive here, there was probably a state with index -1
        # we try to fix it by ignoring transitions in and out of those states
        # (we set both the count and the indices for those transitions to 0)
        mask = transitions < 0
        counts[mask[0,:] | mask[1,:]] = 0
        transitions[mask] = 0
        C = scipy.sparse.coo_matrix((counts, transitions),shape=(numstates,numstates))
            
    return C


def EstimateRateMatrix(tCount, Assignments):
    """ Gives the maximum likelihood estimate for a rate matrix given the data
        Input:  tCount, the matrix of observed counts
        Output: K, most likely rate matrix """

    # Find the estimated dwell times (need to deal w negative ones)
    neg_ind = np.where( Assignments == -1)
    n = np.max( Assignments.flatten() ) + 1
    Assignments[neg_ind] = n
    R = np.bincount( Assignments.flatten() )
    R = R[:n]
    assert tCount.shape[0] == R.shape[0]

    # Most Likely Estimator ( Kij(hat) = Nij / Ri )
    if scipy.sparse.isspmatrix(tCount):
        C = scipy.sparse.csr_matrix( tCount ).asfptype()
        D = scipy.sparse.dia_matrix( (1./R,0), C.shape).tocsr()
        K = D*C # if all is sparse is matrix multiply, formerly: D.dot( C )
    else:
        # deprecated due to laziness --TJL
        print "ERROR! Pass sparse matrix to EstimateRateMatrix()"
        sys.exit(1)
   
    # Now get the diagonals right. They should be negative row sums
    row_sums = np.asarray( C.sum(axis=1) ).flatten()
    current  = K.diagonal()
    S = scipy.sparse.dia_matrix( ((row_sums + (2.0*current)), 0), C.shape).tocsr()
    K = K - S
    assert K.shape == tCount.shape
    assert K.sum(0).all() == np.zeros(K.shape[0]).all()
 
    return K

def EstimateTransitionMatrix(tCount):
    """Simple Maximum Likelihood estimator of transition matrix.
    
    Inputs
    ----------
    tCount : array / sparse matrix
        A square matrix of transition counts
    
    MakeSymmetric : bool
        If true, make transition count matrix symmetric
    
    Returns:
    -------
    tProb : array / sparse matrix
        Estimate of transition probability matrix
    
    Notes:
    -----
    The transition count matrix will not be altered by this function. Its elemnts can
    be either of integer of floating point type.
    """
    #1.  Make sure you don't modify tCounts.
    #2.  Make sure you handle both floats and ints 
    if scipy.sparse.isspmatrix(tCount):
        C=scipy.sparse.csr_matrix(tCount).asfptype()
        weights = np.asarray(C.sum(axis=1)).flatten()            
        D=scipy.sparse.dia_matrix((1./weights,0),C.shape).tocsr()
        tProb=D.dot(C)
    else:
        tProb = np.asarray(tCount.astype(float))                              # astype creates a copy, so tProb is decoupled from tCont
        weights = tProb.sum(axis=1)
        tProb = tProb / weights.reshape((weights.shape[0],1))

    return tProb

def CheckForBadEigenvalues(Eigenvalues,decimal=5,CutoffValue=0.999999):
    """Having multiple eigenvalues of lambda>=1 suggests either non-ergodicity or numerical error.  Throw an error in such cases."""

    if DisableErrorChecking:
        return

    if abs( Eigenvalues[0] - 1 ) > 1-CutoffValue:
        print """WARNING: the largest eigenvalue is not 1, suggesting numerical error.  Try using 64 or 128 bit precision."""

        if Eigenvalues[1] > CutoffValue:
            print """WARNING: the second largest eigenvalue (x) is close to 1, suggesting numerical error or nonergodicity.  Try using 64 or 128 bit precision.  Your data may also be disconnected, in which case you cannot simultaneously model both disconnected components.  Try collecting more data or trimming the disconnected pieces."""
        
def GetEigenvectors(T,NumEig,Epsilon=.001,DenseCutoff=50):
    """Return the left eigenvectors of a transition matrix.  Return sorted by eigenvalue magnitude.

    Inputs:
    T -- A transition matrix.  If T is sparse, sparse eigensolvers will be used.
    NumEig --  How many eigenvalues to calculate and return.

    Keyword Arguments:
    Epsilon -- Throw error if T is not a stochastic matrix, with tolerance given by Epsilon.  Default: 0.001

    Notes:
    Vectors are returned in columns of matrix.
    """
    CheckTransition(T,Epsilon=Epsilon)
    CheckDimensions(T)
    n=T.shape[0]
    if NumEig>n:
        raise Exception("You cannot calculate %d Eigenvectors from a %d x %d matrix"%(NumEig,n,n))
    if n < DenseCutoff and scipy.sparse.issparse(T):
        T=T.toarray()
    if scipy.sparse.issparse(T):
        eigSolution = sparseEigen(T.transpose().tocsr(), max(NumEig,MinimumAllowedNumEig), which="LR", maxiter=100000)
    else:
        eigSolution=eig(T.transpose())      
    Ord=np.argsort(-np.real(eigSolution[0]))

    elambda=eigSolution[0][Ord]
    eV=eigSolution[1][:,Ord]
    
    CheckForBadEigenvalues(elambda, CutoffValue=1-Epsilon) # this is bad IMO --TJL

    eV[:,0]/=sum(eV[:,0])
    eigSolution=(elambda[0:NumEig],eV[:,0:NumEig])
    
    return(eigSolution)

def GetEigenvectors_Right(T,NumEig,Epsilon=.001):
    """Return the right eigenvectors of a transition matrix.  Return sorted by eigenvalue magnitude.

    Inputs:
    T -- A transition matrix.  If T is sparse, sparse eigensolvers will be used.
    NumEig --  How many eigenvalues to calculate and return.

    Keyword Arguments:
    Epsilon -- Throw error if T is not a stochastic matrix, with tolerance given by Epsilon.  Default: 0.001

    Notes:
    Vectors are returned in columns of matrix.
    """
    
    CheckTransition(T,Epsilon=Epsilon)
    CheckDimensions(T)
    n=T.shape[0]
    if NumEig>n:
        raise Exception("You cannot calculate %d Eigenvectors from a %d x %d matrix"%(NumEig,n,n))
    if scipy.sparse.issparse(T):
        eigSolution = sparseEigen(T.tocsr(), max(NumEig,MinimumAllowedNumEig), which="LR", maxiter=100000)
    else:
        eigSolution=eig(T)
    Ord=np.argsort(-np.real(eigSolution[0]))
    elambda=eigSolution[0][Ord]
    eV=eigSolution[1][:,Ord]
                   
    eV[:,0]/=sum(eV[:,0])
    eigSolution=(elambda[0:NumEig],eV[:,0:NumEig])
    
    return(eigSolution)

def GetImpliedTimescales(AssignmentsFn, NumStates, LagTimes, NumImpliedTimes=100, Slide=True, Trim=True, Symmetrize=None, nProc=1):
    """Calculate implied timescales in parallel using multiprocessing library.  Does not work in interactive mode."""
    pool = multiprocessing.Pool(processes=nProc)
    n = len(LagTimes)
    inputs = zip(n*[AssignmentsFn], n*[NumStates], LagTimes, n*[NumImpliedTimes], n*[Slide], n*[Trim], n*[Symmetrize])
    #result = pool.map_async(GetImpliedTimescalesHelper, inputs)
    result = pool.map_async(GetImpliedTimescalesHelper, inputs)
    #result.get(9999999)
    #result.wait()
    lags = result.get(999999)

    # reformat
    formatedLags = np.zeros((n*NumImpliedTimes, 2))
    i = 0
    for arr in lags:
        formatedLags[i:i+NumImpliedTimes,0] = arr[0]
        formatedLags[i:i+NumImpliedTimes,1] = arr[1]
        i += NumImpliedTimes
    return formatedLags

def GetImpliedTimescalesHelper(args):
    """Helper Function.  Calculate implied timescales in parallel using multiprocessing library.  Does not work in interactive mode."""
    AssignmentsFn = args[0]
    NumStates = args[1]
    LagTime = args[2]
    NumImpliedTimes = args[3]
    Slide = args[4]
    Trim = args[5]
    Symmetrize = args[6]

    Assignments=Serializer.LoadData(AssignmentsFn)

    Counts=GetCountMatrixFromAssignments(Assignments,NumStates,LagTime=LagTime,Slide=Slide)
        
    # Apply ergodic trim if requested
    if Trim: Counts, MAP = ErgodicTrim(Counts) # TJL 5/9/11, previously AD

    # Apply a symmetrization scheme (TJL 8/3/11)
    if Symmetrize == 'MLE':
            Counts = IterativeDetailedBalance(Counts, Prior=0.0)
    elif Symmetrize == 'MLE-TNC':
        Counts = EstimateReversibleCountMatrix(Counts, Prior=0.0)       
    elif Symmetrize == 'Transpose':
        Counts = 0.5*(Counts + Counts.transpose())
    elif Symmetrize == None:
        pass
    else:
        print "ERROR: Invalid symmetrization scheme requested: %d. Exiting." %Symmetrize
        sys.exit(1)

    # Calculate the eigen problem
    T=EstimateTransitionMatrix(Counts)
    EigAns=GetEigenvectors(T,NumImpliedTimes+1,Epsilon=1) #TJL: set Epsilon high, should not raise err here     

    # make sure to leave off equilibrium distribution
    lagTimes = LagTime*np.ones((NumImpliedTimes))
    impTimes = -lagTimes/np.log(EigAns[0][1:NumImpliedTimes+1])

    # save intermediate result in case of failure
    res = np.zeros((NumImpliedTimes, 2))
    res[:,0] = lagTimes
    res[:,1] = impTimes

    return (lagTimes, impTimes)

def Sample(T,State,Steps,Traj=None,ForceDense=False):
    """Generate a trajectory of states by propogating a transition matrix.

    Inputs:
    T -- A transition matrix. 
    State -- Starting state for trajectory.
             If State is an integer, it will be used as the initial state.
         If State is None, an initial state will be randomly chosen from an uniform distribution.
         If State is an array, it represents a probability distribution from which the initial
           state will be drawn.
             If a trajectory is specified (see Traj keyword), this variable will be ignored, and the last
           state of that trajectory will be used.
    Steps -- How many steps to generate.
    
    Keyword Arguments:
    Traj -- An existing trajectory (python list) can be input; results will be appended to it.  Default: None
    ForceDense -- Force dense arithmatic.  Can speed up results for small models (OBSOLETE).
    """

    CheckTransition(T)
    CheckDimensions(T)

    if scipy.sparse.isspmatrix(T):
        T = T.tocsr()
        
    # reserve room for the new trajectory (will be appended to an existing trajectory at the end if necessary)
    newtraj = [-1] * Steps

    # determine initial state
    if Traj is None or len(Traj) == 0:
        if State is None:
            State = np.random.randint(T.shape[0])
        elif isinstance(State,np.ndarray):
            State = np.where(scipy.random.multinomial(1,State/sum(State))==1)[0][0]
        newtraj[0] = State
        start = 1
    else:
        State = Traj[-1]
        start = 0
    assert State < T.shape[0], "Intial state is " + str(State) + ", but should be between 0 and " + str(T.shape[0]-1) + "."

    # sample the Markov chain
    if isinstance(T,np.ndarray):
        for i in xrange(start,Steps):
            p = T[State,:]
            State = np.where(scipy.random.multinomial(1,p) == 1)[0][0]
            newtraj[i] = State
    elif isinstance(T, scipy.sparse.csr_matrix):
        if ForceDense:
            # Lutz: this is the old code path that converts the row of transition probabilities to a dense array at each step.
            # With the optimized handling of sparse matrices below, this can probably be deleted altogether.
            for i in xrange(start,Steps):
                p = T[State,:].toarray().flatten()
                State = np.where(scipy.random.multinomial(1,p) == 1)[0][0]
                newtraj[i] = State
        else:
            for i in xrange(start,Steps):
                # Lutz: slicing sparse matrices is very slow (compared to slicing ndarrays)
                # To avoid slicing, use the underlying data structures of the CSR format directly
                vals = T.indices[T.indptr[State]:T.indptr[State+1]]   # the column indices of the non-zero entries are the possible target states
                p = T.data[T.indptr[State]:T.indptr[State+1]]         # the data values of the non-zero entries are the corresponding probabilities
                State = vals[np.where(scipy.random.multinomial(1,p) == 1)[0][0]]
                newtraj[i] = State
    else:
        raise RuntimeError, "Unknown matrix type: " + str(type(T))

    # return the new trajectory, or the concatenation of the old trajectory with the new one
    if Traj is None:
        return newtraj
    else:
        Traj.extend(newtraj)
        return Traj

def PropagateModel(T,NumSteps,X0,ObservableVector=None):
    """Propogate the time evolution of a population vector.

    Inputs:
    T -- A transition matrix.  
    NumSteps -- How many timesteps to iterate.
    X0 -- The initial population vector.
    
    Keyword Arguments:
    ObservableVector -- a vector containing the state-wise averaged property of some observable.  Can be used to propagate properties such as fraction folded, ensemble average RMSD, etc.  Default: None
    """
    CheckTransition(T)
    if ObservableVector==None:
        CheckDimensions(T,X0)
    else:
        CheckDimensions(T,X0,ObservableVector)

    X=X0.copy()
    obslist=[]
    if scipy.sparse.issparse(T):
        TC=T.tocsr()
    else:
        TC=T
    Tl=scipy.sparse.linalg.aslinearoperator(TC)
    for i in range(NumSteps):
        X=Tl.rmatvec(X);
        if ObservableVector!=None:
            obslist.append(sum(ObservableVector*X))
    return X,obslist

def GetCountMatrixFromAssignments(Assignments,NumStates=None,LagTime=1,Slide=True):
    """Calculate count matrix from Assignments.

    Inputs:
    Assignments -- a numpy array containing the state assignments.  

    Keyword Arguments:
    NumStates -- Can be automatically determined, unless you want a model with more states than are observed.  Default: None
    LagTime -- the LagTime with which to estimate the count matrix. Default: 1
    Slide -- Use a sliding window.  Default: True

    Notes:
    Assignments are input as iterables over numpy 1-d arrays of integers.
    For example a 2-d array where Assignments[i,j] gives the ith trajectory, jth frame.
    The beginning and end of each trajectory may be padded with negative ones, which will be ignored.
    If the number of states is not given explitly, it will be determined as one plus the largest state index of the Assignments.
    Sliding window yields non-independent samples, but wastes less data.
    """

    if not NumStates:
        NumStates = 1 + int(np.max([np.max(a) for a in Assignments]))   # Lutz: a single np.max is not enough, b/c it can't handle a list of 1-d arrays of different lengths
        assert NumStates >= 1
        
    C=scipy.sparse.lil_matrix((int(NumStates),int(NumStates)),dtype='float32')  # Lutz: why are we using float for count matrices?
    
    for A in Assignments:
        FirstEntry=np.where(A!=-1)[0]
        if len(FirstEntry)>=1:#New Code by KAB to skip pre-padded negative ones.  This should solve issues with Tarjan trimming results.
            FirstEntry=FirstEntry[0]
            A=A[FirstEntry:]
            C=C+GetTransitionCountMatrixSparse(A,NumStates,LagTime=LagTime,slidingwindow=Slide)#.tolil()
    return(C)

def GetDistributionsOverTime(Ass,n):
    """Calculate raw populations as a function of time; used  to compare raw and MSM populations."""
    n1,n2=Ass.shape
    X=np.zeros((n2,n),dtype='float32')
    for i in range(n2):
        print(i)
        for j in range(n1):
            x=Ass[j,i]
            if x > -1:
                X[i,x]+=1.
                X[i]=X[i]/sum(X[i])
    return(X)


def ApplyMappingToAssignments(Assignments,Mapping):
    """Remap the states in an assignments file according to a mapping.  Useful after performing PCCA or Ergodic Trimming."""
    A=Assignments
    
    NewMapping=Mapping.copy()
    NewMapping[np.where(Mapping==-1)]=Mapping.max()+1#Make a special state for things that get deleted by Ergodic Trimming.

    NegativeOneStates=np.where(A==-1)
    A[:]=NewMapping[A]
    WhereEliminatedStates=np.where(A==(Mapping.max()+1))

    A[NegativeOneStates]=-1#These are the dangling 'tails' of trajectories (with no actual data) that we denote state -1.
    A[WhereEliminatedStates]=-1#These states have typically been "deleted" by the ergodic trimming algorithm.  Can be at beginning or end of trajectory.

def ApplyMappingToVector(V, Mapping):
        """ Applys the mapping to an observable vector. This is a helper function mostly for documentation and completeness """
        NV = V[np.where(Mapping != -1)[0]] 
        print "Mapping %d elements --> %d" % (len(V), len(NV))
        return NV

def RenumberStates_SLOW(Assignments):
    """Renumber states to be consecutive integers (0, 1, ... , n); useful if some states have 0 counts."""
    A=Assignments
    GoodStates=np.unique(A)
    if GoodStates[0]==-1: GoodStates=GoodStates[1:]
    MinusOne=np.where(A==-1)
    for i,x in enumerate(GoodStates):
        A[np.where(A==x)]=i
    A[MinusOne]=-1


def RenumberStates(Assignments):
    """Renumber states to be consecutive integers (0, 1, ... , n);
    useful if some states have 0 counts.
    
    Should do the same thing as RenumberStates_SLOW() without using np.where()
    repeatedly, which can be slow"""
    
    unique = list(np.unique(Assignments))
    if unique[0] == -1:
        minus_one = np.where(Assignments == -1)
        unique.pop(0)
    else:
        minus_one = []
    
    inverse_mapping = defaultdict(lambda: ([], []))
    for i in xrange(Assignments.shape[0]):
        for j in xrange(Assignments.shape[1]):
            inverse_mapping[Assignments[i,j]][0].append(i)
            inverse_mapping[Assignments[i,j]][1].append(j)
    
    for i,x in enumerate(unique):
        Assignments[inverse_mapping[x]] = i
    Assignments[minus_one] = -1


def TrimHighRMSDToCenters(Ass,RMSD,Epsilon=.25):
    """Null out Assignments where RMSD to cluster center is too high.  If RMSD to cluster center is too high, then we expect large kinetic barriers within a state.  """
    r=RMSD
    x0,x1=np.where(r>Epsilon)
    WhichTrajs=np.unique(x0)
    for i in WhichTrajs:
        y=min(x1[np.where(x0==i)])
        Ass[i,y:]=-1

def Tarjan(graph):
    """ Find the strongly connected components in a graph using Tarjan's algorithm.
    
    Inputs:
    graph  -- a dictionary mapping node names to lists of successor nodes.

    Notes:
    Code based on ActiveState code by Josiah Carlson (New BSD license).
    Most users will want to call the ErgodicTrim() function rather than directly calling Tarjan().
        """
    NumStates=graph.shape[0]

    #Keeping track of recursion state info by node
    Nodes=np.arange(NumStates)
    NodeNums=[None for i in range(NumStates)]
    NodeRoots=np.arange(NumStates)
    NodeVisited=[False for i in range(NumStates)]
    NodeHidden=[False for i in range(NumStates)]
    NodeInComponent=[None for i in range(NumStates)]

    stack = []
    components = []
    nodes_visit_order = []
    graph.next_visit_num = 0

    def visit(v):
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
                    NodeRoots[v] = nodes_visit_order[ min(NodeNums[NodeRoots[v]],\
                                                     NodeNums[NodeRoots[w]])]
            cont = 0
            for w in iterator:
                if not NodeVisited[w]:
                    cont = 1
                    call_stack.append((0, v, iterator, w))
                    call_stack.append((1, w, graph.getrow(w).nonzero()[1], None))
                    break
                if not NodeInComponent[w]:
                    NodeRoots[v] = nodes_visit_order[ min(NodeNums[NodeRoots[v]],\
                                                         NodeNums[NodeRoots[w]]) ]
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

def ErgodicTrim(Counts,Assignments=None):
    """Use Tarjan's Algorithm to find maximal strongly connected subgraph.
    
    Inputs:
    Counts -- sparse matrix of counts.

    Keywoard Arguments:
    Assignments -- Optionally map assignments to the new states, nulling out disconnected regions.

    Notes:
    The component with maximum number of counts is used.
    """
    
    NZ=np.array(Counts.nonzero()).transpose()

    ConnectedComponents=Tarjan(Counts)
    PiSym=np.array(Counts.sum(0)).flatten()
    ComponentPops=np.array([sum(PiSym[np.array(x)]) for x in ConnectedComponents])
    ComponentInd=np.argmax(ComponentPops)
    print("Selected component %d with population %f"%(ComponentInd,ComponentPops[ComponentInd]/ComponentPops.sum()))
    GoodComponent=np.unique(ConnectedComponents[ComponentInd])

    Mapping=np.zeros(Counts.shape[0],dtype='int')-1
    for i,x in enumerate(GoodComponent):
        Mapping[x]=i

    NZ[:,0]=Mapping[NZ[:,0]]
    NZ[:,1]=Mapping[NZ[:,1]]

    Ind=np.where(NZ.min(1)!=-1)
    X=scipy.sparse.csr_matrix((Counts.data[Ind],NZ[Ind].transpose()))

    if Assignments!=None:
        ApplyMappingToAssignments(Assignments,Mapping)
    
    return(X,Mapping)

def UpdateAssignmentsAndCounts(Assignments,Counts,D,XS,NewState,State):
    """Updates the count matrix, self-transition probabilities, row sums, and assignments after lumping State into NewState.  REQUIRES SYMMETRIC COUNTS. Used in EnforceMetastability and EnforceCounts."""

    Assignments[np.where(Assignments==State)]=NewState

    Counts[NewState,NewState]+=(Counts[State,State]+2*Counts[NewState,State])
    Counts[State,State]=0.
    Counts[State,NewState]=0.
    Counts[NewState,State]=0.

    nz=np.array(Counts[State,:].nonzero()).transpose()
    for i,(x,y) in enumerate(nz):
        if y!=NewState:
            Counts[NewState,y]+=Counts[State,y]
            Counts[y,NewState]+=Counts[State,y]
            Counts[State,y]=0.
            Counts[y,State]=0.

    XS[NewState]=XS[State]+XS[NewState]
    XS[State]=100000000.0

    D[State]=1.
    D[NewState]=Counts[NewState,NewState]/XS[NewState]
    
def EnforceMetastability(Assignments,DesiredNumStates,LagTime=1):
    """Merge states that are highly unstable to ensure that all states have self-transition probabilities of 0.5 or greater.  Merging occurs only between neighbors.  States are chosen to be merged in order of metastability."""

    NumStates=max(Assignments.flatten())+1
    NumMergeAttempts=NumStates-DesiredNumStates
    Counts=GetCountMatrixFromAssignments(Assignments,NumStates,LagTime=LagTime,Slide=True)
    Counts=Counts+Counts.transpose()
    
    XS=np.array((Counts).sum(1)).flatten()
    D=Counts.diagonal()/XS
    D[np.where(XS==0.)]=1.

    for CounterInt in xrange(NumMergeAttempts):

        i=np.argmin(D)

        Neighbors=np.array(Counts[:,i].nonzero())[0]

        M=np.zeros(Neighbors.shape)
        for k, j in enumerate(Neighbors):
            if j!=i:
                M[k]=(Counts[i,i]+Counts[j,j]+2*Counts[i,j])/(XS[i]+XS[j])
        j=Neighbors[np.argmax(M)]#Pick the state to lump by seeing which one results in the most metastable lumped state.  Heuristic #1        
        print("Iter %d.  Merging States %d,%d.  Old Metastabilities: %f, %f."%(CounterInt,i,j,D[i],D[j]))
        UpdateAssignmentsAndCounts(Assignments,Counts,D,XS,i,j)
        
    RenumberStates(Assignments)

def EnforceCounts(Assignments,LagTime=1,MinCounts=3):
    """Merge states with few counts to improve statistics.

    Notes:
    1.  Find the state with lowest counts.
    2.  Determine its neighbors.
    3.  Lump state with neighbor that it is most connected to.

    By reducing the number of states, this can improve statistics and avoid negative eigenvalues.
    """

    NumStates=max(Assignments.flatten())+1
    Counts=GetCountMatrixFromAssignments(Assignments,NumStates,LagTime=LagTime,Slide=True)
    Counts=0.5*(Counts+Counts.transpose())
    Counts=Counts.tolil()
    X=np.array((Counts).sum(1)).flatten()
    
    D=Counts.diagonal()/X
    D[np.where(X==0.)]=1.

    k=0
    while True:
        k+=1
        i=np.argmin(X)
        if X[i] >= MinCounts:
            break
        Neighbors=np.array(Counts[:,i].nonzero())[0]
        NbrCounts=np.zeros(Neighbors.shape)
        for L,n in enumerate(Neighbors):
            if n!=i:
                NbrCounts[L]=Counts[n,i]
            
        XNb=X[Neighbors].copy()
        XNb[np.where(Neighbors==i)]=10000000.

        #Find the neighbor with most counts betwen i and j
        try:
            Relativej=np.argmax(NbrCounts)
        except ValueError:#This means that NbrCounts is empty.  Thus, state has NO neighbors.
            print("Warning: state %d was completely empty."%i)
            Counts[i,i]=100000000.
            continue

        #Next, we look at ALL neighbors with the same number of counts  Cij as the maximum (e.g. just as kinetically related as relativej).
        EquivalentRelativej=np.where(NbrCounts==NbrCounts[Relativej])[0]

        #We pick j to be neighbor with the Maximal connectivity but the poorest statistics.  The statistics filter is applied only as a "tiebreaker"
        #to ensure that if possible, states that already have good statistics do NOT get things lumped into them.  
        RelativeRelativej=np.argmin(XNb[EquivalentRelativej])
        Relativej=EquivalentRelativej[RelativeRelativej]
        
        j=Neighbors[Relativej]
        print("Iter %d.  Merging States %d,%d., %d %d"%(k,i,j,X[i],X[j]))
        if X[i]+X[j] > 10000.:
            Counts=Counts.tolil()
        UpdateAssignmentsAndCounts(Assignments,Counts,D,X,i,j)
        if X[i]+X[j] > 10000.:
            Counts=Counts.tocsr()
        
    RenumberStates(Assignments)
    
def IterativeDetailedBalanceWithPrior(Counts,Alpha,NumIter=1000):
    """Use MLE to Estimate symmetric (e.g. reversible) count matrix from the unsymmetric counts.

    Inputs:
    Counts -- Sparse CSR matrix of counts.

    Keyword Arguments:
    NumIter -- Maximum number of iterations.  Default: 10000000
    TerminationEpsilon -- Terminate when |Pi^{k+1}-Pi^{k}| < epsilon.  Default: 1E-10
    Prior -- Add prior counts to EVERY transition. 

    Notes:
    1.  Also known as the Boxer method.
    2.  This tends to be very slow, due to the calculations using the prior counts.
    3.  The prior is handled implicitly during calculations, so no dense matrices are stored.
    4.  ReconstructDense() can be used to reconstruct a dense matrix using the results of this.
    
    """
    NumStates=Counts.shape[0]

    S=Counts+Counts.transpose()
    N=np.array(Counts.sum(1)).flatten()
    Na=N+NumStates*Alpha

    NS=np.array(S.sum(1)).flatten()
    NS/=NS.sum()
    Ind=np.argmax(NS)

    NZX,NZY=np.array(S.nonzero())

    Q=S.copy()
    XS=np.array(Q.sum(0)).flatten()

    for k in xrange(NumIter):
        print(k,NS[Ind],XS[Ind])
        V=Na/XS
        Q.data[:]=S.data/(V[NZX]+V[NZY])

        RS=np.zeros(NumStates)
        for a in xrange(NumStates):
            RS+=1./(V+V[a])

        RS*=2.*Alpha
        QS=np.array(Q.sum(0)).flatten()

        XS=RS+QS
        XS/=XS.sum()
        
    return(Q,V,XS)

def ReconstructDense(Q,V,Alpha):
    """Reconstruct a dense count matrix from output of IterativeDetailedBalanceWithPrior.

    Notes:
    1.  After you reconstruct the dense matrix, you could possible re-sparsify it by discarding all counts less than some threshold, e.g.

    X[where(X<Epsilon)]=0.
    X=scipy.sparse.csr_matrix(X)
    """
    
    X=Q.toarray()
    NumStates=X.shape[0]
    for i in xrange(NumStates):
        for j in xrange(NumStates):
            X[i,j]+=2*Alpha/(V[i]+V[j])
    return(X)

def IterativeDetailedBalance(Counts,NumIter=10000000,TerminationEpsilon=1E-10,Prior=0.):
    """Use MLE to Estimate symmetric (e.g. reversible) count matrix from the unsymmetric counts.

    Inputs:
    Counts -- Sparse CSR matrix of counts.

    Keyword Arguments:
    NumIter -- Maximum number of iterations.  Default: 10000000
    TerminationEpsilon -- Terminate when |Pi^{k+1}-Pi^{k}| < epsilon.  Default: 1E-10
    Prior -- Add prior counts to stencil of symmetrized counts.  

    Notes:
    1.  Also known as the Boxer method.
    2.  Requires strongly ergodic counts as input, otherwise will simply focus on sink.
    """
    # make sure that count matrix is properly formatted, sparse CSR matrix without zeros
    if not scipy.sparse.isspmatrix(Counts):
        Counts = scipy.sparse.csr_matrix(Counts)
    Counts = Counts.asformat("csr").asfptype()
    Counts.eliminate_zeros()
    
    if (Prior is not None) and (Prior != 0):
        PriorMatrix=(Counts+Counts.transpose()).tocsr()
        PriorMatrix.data*=0.
        PriorMatrix.data+=Prior
        Counts=Counts+PriorMatrix
        print("Added prior value of %f to count matrix" % Prior)
    
    S=Counts+Counts.transpose()
    N=np.array(Counts.sum(1)).flatten()
    Na=N

    NS=np.array(S.sum(1)).flatten()
    NS/=NS.sum()
    Ind=np.argmax(NS)

    NZX,NZY=np.array(S.nonzero())

    Q=S.copy()
    XS=np.array(Q.sum(0)).flatten()

    for k in xrange(NumIter):
        Old=XS
        V=Na/XS
        Q.data[:]=S.data/(V[NZX]+V[NZY])
        QS=np.array(Q.sum(0)).flatten()
        
        XS=QS
        XS/=XS.sum()
        PiDiffNorm=np.linalg.norm(XS-Old)
        print(k,NS[Ind],XS[Ind],PiDiffNorm)
        if PiDiffNorm< TerminationEpsilon:
            break

    Q/=Q.sum()
    Q*=Counts.sum()
    return(Q)

def EnforceTrimEstimate(Assignments,LagTime=1,MinCounts=0,Slide=True,BoxerIter=1000000,Prior=0.0):
    """Enforce a minimum number of counts, trim the data, and estimate transition matrix.

    Inputs:
    Assignments -- A numpy matrix of assignments

    Keyword Arguments:
    LagTime -- The LagTime used to estimate Count matrix.  Default: 0
    MinCounts -- Enforce a minimum number of counts in each state.  Default: 0
    Slide -- Use sliding window when estimating counts.  Default: True
    BoxerIter -- Maximum number of iterations when calculating MLE.  Default: 1000000
    Prior -- Add Prior counts in stencil of symmetrized counts.  Default: 0

    Notes:
    1.  Merge states with low counts to ensure good statistics.
    2.  Apply Tarjan's algorithm to find the maximal strongly connected subgraph.
    3.  Use reversible MLE estimator to estimate a normalized symmetric count matrix.
    Operates in place (destructively) on the Assignments input.
    """

    EnforceCounts(Assignments,LagTime=LagTime,MinCounts=MinCounts)

    NumStates=max(Assignments.flatten())+1
    Counts=GetCountMatrixFromAssignments(Assignments,NumStates,LagTime=LagTime,Slide=Slide)
    NumStates=Counts.shape[0]
    Counts=Counts

    UnSymmetrizedCounts,Mapping=ErgodicTrim(Counts)
        
    ApplyMappingToAssignments(Assignments,Mapping)
    
    ReversibleCounts=IterativeDetailedBalance(UnSymmetrizedCounts,NumIter=BoxerIter,Prior=Prior)

    return(ReversibleCounts,UnSymmetrizedCounts,Mapping)

def logLikelihood(C, P):
    """Returns the log of the likelihood of an observed count matrix C given a transition matrix P.

    Arguments
    ---------
    C : array or sparse matrix
    Transition count matrix.
    P : array or sparse matrix
    Transition probability matrix.

    Returns
    -------
    The natural log of the likelihood, computed as sum_ij C_ij log(P_ij)."""

    if isinstance(P, np.ndarray) and isinstance(C, np.ndarray):
        C = np.asarray(C)   # make sure that both C and P are arrays
        P = np.asarray(P)   # (not dense matrices), so we can use element-wise multiplication
        mask = C > 0
        return np.sum( np.log(P[mask]) * C[mask] )
    else:
        # make sure both C and P are sparse CSR matrices
        if not scipy.sparse.isspmatrix(C):
            C = scipy.sparse.csr_matrix(C)
        else:
            C = C.tocsr()
        if not scipy.sparse.isspmatrix(P):
            P = scipy.sparse.csr_matrix(P)
        else:
            P = P.tocsr()
        row, col = C.nonzero()
        return np.sum( np.log(np.asarray(P[row,col])) * np.asarray(C[row,col]) )

def EstimateReversibleCountMatrix(C, Prior=0., InitialGuess = None):
    """Calculates the maximum-likelihood symmetric count matrix for a givnen observed count matrix.

    This function uses a Newton conjugate-gradient algorithm to maximize the likelihood
    of a reversible transition probability matrix.
    
    Arguments
    ---------
    C : array or sparse matrix
        Transition count matrix.
    Prior : float
        If not zero, add this value to the count matrix for every transition
    that has occured in either direction.
    InitialGuess : array or sparse matrix
        Initial guess for the symmetric count matrix uses as starting point for
    likelihood maximization. If None, the naive symmetrized guess 0.5*(C+C.T)
    is used.

    Returns
    -------
    Symmetric count matrix. If C is sparse then the returned matrix is also sparse, and
    dense otherwise.

    See also
    --------
    IterativeDetailedBalance
    """

    def negativeLogLikelihoodFromCountEstimatesSparse(Xupdata,row,col,N,C):
        """Calculates the negative log likelihood that a symmetric count matrix X gave
    rise to an observed transition count matrix C, as well as the gradient
    d -log L / d X_ij."""

        assert np.alltrue(Xupdata > 0)

        Xup = scipy.sparse.csr_matrix((Xupdata, (row,col)), shape=(N,N))                    # Xup is the upper triagonal (inluding the main diagonal) of the symmetric count matrix
        X = Xup + Xup.T - scipy.sparse.spdiags(Xup.diagonal(),0,Xup.shape[0],Xup.shape[1])  # X is the complete symmetric count matrix
        Xs = np.array(X.sum(axis=1)).ravel()                                                # Xs is the array of row sums of X: Xs_i = sum_j X_ij
        XsInv = scipy.sparse.spdiags(1./Xs, 0, len(Xs), len(Xs))
        P = (XsInv * X).tocsr()                                                             # P is now the matrix P_ij = X_ij / sum_j X_ij
        logP = scipy.sparse.csr_matrix((np.log(P.data),P.indices,P.indptr))
        logL = np.sum(C.multiply(logP).data)                                                # logL is the log of the likelihood: sum_ij C_ij log(X_ij / Xs_i)

        Cs = np.array(C.sum(axis=1)).ravel()                                                # Cs is the array of row sums of C: Cs_i = sum_j C_ij
        srow, scol = X.nonzero()                                                            # remember the postitions of the non-zero elements of X
        Udata = np.array((C[srow,scol] / X[srow,scol]) - (Cs[srow]/Xs[srow])).ravel()       # calculate the derivative: d(log L)/dX_ij = C_ij/X_ij - Cs_i/Xs_i
        U = scipy.sparse.csr_matrix((Udata,(srow,scol)),shape=(N,N))                        # U is the matrix U_ij = d(log L) / dX_ij

        # so far, we have assumed that all the partial derivatives wrt. X_ij are independent
        # however, the degrees of freedom are only X_ij for i <= j
        # for i != j, the total change in log L is d(log L)/dX_ij + d(log L)/dX_ji

        gradient = (U + U.T - scipy.sparse.spdiags(U.diagonal(),0,U.shape[0],U.shape[1]) ).tocsr()

        # now we have to convert the non-zero elements of the upper triangle into the
        # same 1-d array structure that was used for Xupdata

        gradient = np.array(gradient[row,col]).reshape(-1)

    #   print  "max g:", np.max(gradient), "min g:", np.min(gradient), "|g|^2", (gradient*gradient).sum(), "g * X", (gradient*Xupdata).sum()
        return -logL, -gradient

        # current implementation only for sparse matrices
        # if given a dense matrix, sparsify it, and turn the result back to a dense array
        if not scipy.sparse.isspmatrix(C):
            return EstimateReversibleCountMatrix(scipy.sparse.csr_matrix(C), Prior=Prior, InitialGuess=InitialGuess).toarray()

        N = C.shape[0]
        assert C.shape[1] == N, "Count matrix is not square, but has shape " + str(C.shape)

        C = C.tocsr()
        C.eliminate_zeros()
        if (Prior is not None) and (Prior != 0):        # add prior if necessary
            PriorMatrix=(C+C.transpose()).tocsr()
            PriorMatrix.data*=0.
            PriorMatrix.data+=Prior
            C=C+PriorMatrix
            print("Added prior value of %f to count matrix" % Prior)

        # initial guess for symmetric count matrix
        if InitialGuess is None:
            X0 = 0.5 * (C + C.T)
        else:
            X0 = scipy.sparse.csr_matrix(0.5 * (InitialGuess + InitialGuess.T))  # this guarantees that the initial guess is indeed symmetric (and sparse)
        initialLogLikelihood = logLikelihood(C, EstimateTransitionMatrix(X0))

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
        print "Log-Likelihood of intial guess for reversible transition probability matrix:", -negative_logL
        while maximizationrun <= 1000:
            # rescale the X_ij so that the magnitude of the gradient is 1
            gtg = (negative_gradient*negative_gradient).sum()
            scalefactor = np.sqrt(gtg / rescale_target)
            Xupdata[:] *= scalefactor

            # now run the minimizer
            Xupdata, nfeval, rc = scipy.optimize.fmin_tnc(negativeLogLikelihoodFromCountEstimatesSparse, Xupdata, args=(row, col, N, C), bounds=bounds, approx_grad=False, maxfun=rescale_every, disp=0)
            totalnumberoffunctionevaluations += nfeval
            negative_logL, negative_gradient = negativeLogLikelihoodFromCountEstimatesSparse(Xupdata, row, col, N, C)
            print "Log-Likelihood after", totalnumberoffunctionevaluations, "function evaluations:", -negative_logL
            if rc in (0,1,2):
                break    # Converged
            elif rc in (3,4):
                pass     # Not converged, keep going
            else:
                raise RuntimeError, "Likelihood maximization caused internal error (code " + str(rc) + "): " + str(scipy.optimize.tnc.RCSTRINGS[rc])
            maximizationrun += 1
        else:
            print "Warning: maximum could not be obtained."
        print  "Result of last maximization run (run " + str(maximizationrun) + "):", scipy.optimize.tnc.RCSTRINGS[rc]

        Xup = scipy.sparse.coo_matrix((Xupdata, (row,col)), shape=(N,N))    
        X = Xup + Xup.T - scipy.sparse.spdiags(Xup.diagonal(),0,Xup.shape[0],Xup.shape[1])    # reconstruct full symmetric matrix from upper triangle part

        finalLogLikelihood = logLikelihood(C, EstimateTransitionMatrix(X))
        print "Log-Likelihood of final reversible transition probability matrix:", finalLogLikelihood
        print "Likelihood ratio:", np.exp(finalLogLikelihood - initialLogLikelihood)

        # some  basic consistency checks
        if not np.alltrue(np.isfinite(X.data)):
            raise RuntimeError, "The obtained symmetrized count matrix is not finite."
        if not np.alltrue(X.data > 0):
            raise RuntimeError, "The obtained symmetrized count matrix is not strictly positive for all observed transitions, the smallest element is " + str(np.min(X.data))

        return X
