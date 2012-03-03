import numpy as np
from Emsmbuilder import MSMLib

def OldPCCA(T,n,Assignments=None,EigCutoff=0.):
    """Create a lumped model using the PCCA algorithm.  

    Inputs:
    T -- A transition matrix.  
    n -- The desired number of states.

    Keyword Arguments:
    dense -- force dense eigensolver.  Default: False
    Assignments -- Optionally map assignments to new states.  Default: None
    EigCutoff -- Optionally stop splitting states when eigenvalue is less than EigCutoff.  Default: 0

    Notes:
    Returns a mapping from the Microstate indices to the Macrostate indices.
    To construct a Macrostate MSM, you then need to map your Assignment data to the new states (e.g. Assignments=MAP[Assignments]).
    """

    eigSolution=MSMLib.GetEigenvectors_Right(T,n)
    n=len(eigSolution[0])
    eigVecs=(eigSolution[1].transpose())
    
    mapMicroToMacro=np.zeros(T.shape[0],'int')
    for curNumMacro in range(1, n):
        if eigSolution[0][curNumMacro]<=EigCutoff:#New Feature: Quit splitting states based on an eigenvalue (e.g. Timescale) cutoff.
            break
        maxSpread = -1       # max spread seen
        maxSpreadState = -1 # state with max spread
        for currState in range(curNumMacro):
            myComponents = eigVecs[curNumMacro][(mapMicroToMacro==currState).flatten()]
            maxComponent = max(myComponents)
            minComponent = min(myComponents)
            spread = maxComponent - minComponent
            if spread > maxSpread:
                maxSpread = spread
                maxSpreadState = currState
                # split the macrostate with the greatest spread.
                # microstates corresponding to components of macrostate eigenvector
                # greater than mean go in new
                # macrostate, rest stay in current macrostate               
        meanComponent = np.mean(eigVecs[curNumMacro][(mapMicroToMacro==maxSpreadState).flatten()])
        newMacrostateIndices = (eigVecs[curNumMacro]-meanComponent >=0.00001)*(mapMicroToMacro==maxSpreadState)
        mapMicroToMacro[newMacrostateIndices] = curNumMacro

    if Assignments!=None:#If an assignments is given, map them to the new states (inplace).
        MSMLib.ApplyMappingToAssignments(Assignments,mapMicroToMacro)
    return(mapMicroToMacro.astype('int'))


def OldPCCA_Simplex(T, n, doMinimization=False):
    """Create a lumped model using the PCCA+ (Simplex) algorithm.  

    Inputs:
    T -- A transition matrix.  
    n -- The desired number of states.

    Keyword Arguments:
    doMinimization -- perform the minimization step in PCCA+ algorithm.  Default: True

    Notes:
    Returns a mapping from the Microstate indices to the Macrostate indices.
    To construct a Macrostate MSM, you then need to map your Assignment data to the new states (e.g. Assignments=MAP[Assignments]).
    """
    raise Exception("This is broken!")
    nEigPerron = n-1
    eigSolution=MSMLib.GetEigenvectors_Right(T,nEigPerron+1)
    #eigVecs=(eigSolution[1].transpose())
    eigVecs=eigSolution[1]
    eigVals=eigSolution[0]
    mapMicroToMacro=np.zeros(T.shape[0])

        # get eq dist from first left eig vec
    tmp = MSMLib.GetEigenvectors(T, 10) # Get 10 EV to satisfy ARPACK --TJL
    pi=tmp[1][:,0]
    pi /= pi.sum()

    # number eigenvectors
    nEigVecs = np.shape(eigVecs)[1]

    # pi orthogonalization
    for i in range(nEigVecs):
        denom = np.dot(np.transpose(eigVecs[:,nEigVecs-1]*pi), eigVecs[:,nEigVecs-1])
        denom = np.sqrt(denom)
        denom *= np.sign(eigVecs[0,nEigVecs-1])
        eigVecs[:,nEigVecs-1] = eigVecs[:,nEigVecs-1] / denom

    maxScale = 0.0
    maxInd=0
    for i in range(nEigVecs):
        scale = sum(pi * eigVecs[:,i])
        if abs(scale) > maxScale:
            maxScale = abs(scale)
            maxInd = i
    eigVecs[:,maxInd] = eigVecs[:,0]
    eigVecs[:,0] = 1
    eigVecs[np.where(pi<=0)[0],:] = 0

    for i in range(1,nEigVecs):
        for j in range(i-1):
            scale = np.dot(np.transpose(eigVecs[:,j]*pi), eigVecs[:,i])
            eigVecs[:,i] -= scale*eigVecs[:,j]
        sumVal = np.sqrt(np.dot(np.transpose(eigVecs[:,i]*pi), eigVecs[:,i]))
        eigVecs[:,i] /= sumVal

    # find representative microstate for each vertex of simplex (cluster)
    repMicroState = FindSimplexVertices(nEigPerron+1, eigVecs)

    # get initial guess for transformatin matrix A
    A = eigVecs[repMicroState,:]
    print eigVecs.shape
    print repMicroState.shape
    A = np.linalg.inv(A)
    normA = np.linalg.norm(A[1:nEigVecs, 1:nEigVecs])

    # check what min of chi is before optimize
    minChi = (np.dot(eigVecs,A)).min()
    print " Before optimize, chi.min = %f" % minChi

    # get flattened representation of A
    alpha = np.zeros([(nEigVecs-1)*(nEigVecs-1)])
    for i in range(nEigVecs-1):
        for j in range(nEigVecs-1):
            alpha[j + i*(nEigVecs-1)] = A[i+1,j+1]

    # do optimization
    from scipy.optimize import fmin
    initVal = -objectiveFunc(alpha, eigVecs, pi, normA, n)
    print " Initial value of objective function: %f" % initVal
    if doMinimization:
        alpha = fmin(objectiveFunc, alpha, args=(eigVecs, pi, normA, n), maxiter=1e6, maxfun=1e6)
    else:
        print " Skipping Minimization Step"

    # get A back from alpha
    for i in range(nEigVecs-1):
        for j in range(nEigVecs-1):
            A[i+1,j+1] = alpha[j + i*(nEigVecs-1)]

    # fill in missing values in A
    A[1:nEigVecs,0] = -sum(A[1:nEigVecs,1:nEigVecs], 1)
    for j in range(nEigVecs):
        A[0,j] = -np.dot(eigVecs[0,1:nEigVecs], A[1:nEigVecs,j])
        for l in range(1,n):
            dummy = -np.dot(eigVecs[l,1:nEigVecs], A[1:nEigVecs,j])
            if dummy > A[0,j]:
                A[0,j] = dummy
    A /= sum(A[0,:])

    # find chi matrix, membership matrix giving something like probability that each microstate belongs to each vertex/macrostate.
    # say like probability because may get negative numbers and don't necessarily sum to 1 due imperfect simplex structure.
    # rows are microstates, columns are macrostates
    chi = np.dot(eigVecs, A)

    # print final values of things
    minChi = (np.dot(eigVecs,A)).min()
    print " At end, chi.min = %f" % minChi
    finalVal = -objectiveFunc(alpha, eigVecs, pi, normA, n)
    print " Final value of objective function: %f" % finalVal

    # find most probable mapping of all microstates to macrostates.
    mapMicroToMacro = np.argmax(chi,1)
    return(mapMicroToMacro)

def OldobjectiveFunc(alpha, eigVecs, pi, NORMA,nStates):
    """Objective function for PCCA+ algorithm."""
    raise Exception("Broken")
    # number eigenvectors
    nEigVecs = np.shape(eigVecs)[1]

    # get A back from alpha
    A = np.zeros([nEigVecs,nEigVecs])
    for i in range(nEigVecs-1):
        for j in range(nEigVecs-1):
            A[i+1,j+1] = alpha[j + i*(nEigVecs-1)]

    normA = np.linalg.norm(A[1:nEigVecs, 1:nEigVecs])

    # fill in missing values in A
    A[1:nEigVecs,0] = -sum(A[1:nEigVecs,1:nEigVecs], 1)
    for j in range(nEigVecs):
        A[0,j] = -np.dot(eigVecs[0,1:nEigVecs], A[1:nEigVecs,j])
        for l in range(1,nStates):
            dummy = -np.dot(eigVecs[l,1:nEigVecs], A[1:nEigVecs,j])
            if dummy > A[0,j]:
                A[0,j] = dummy
    A /= sum(A[0,:])

    # optimizing trace(S)
    optval = np.trace(np.dot(np.dot(np.diag(1/A[0,:]), np.transpose(A)), A))
    optval = -(optval - (NORMA-normA)*(NORMA-normA))

    return optval

def FindSimplexVertices(nClusters, eigVecs):
    """Find the vertices of the simplex structure.  Do this by finding vectors that are as close as possible to orthogonal.

    Inputs:
    nClusters -- number of Perron clusters (int)
    eigVecs -- first nCluster eigenvectors (matrix of floats).  That is, eigenvectors corresponding to Perron clusters

    Notes:
    Returns list mapping between simplex vertices and representative microstates (microstate that lies exactly on vertex) (array of ints).
    """
    raise Exception("Broken")
    # initialize mapping between simplex verices and microstates
    mapVertToMicro = np.zeros([nClusters], "int32")

    # copy of eigVecs that will use/modify to find orthogonal vectors
    orthoSys = eigVecs.copy()

    # find the first vertex, the eigenvector with the greatest norm (or length).
    # this will be the first of our basis vectors
    maxNorm = 0
    for i in range(np.size(eigVecs,0)):
        dist = np.linalg.norm(eigVecs[i,:])
        if dist > maxNorm:
            maxNorm = dist
            mapVertToMicro[0] = i

    # reduce every row of orthoSys by eigenvector of first vertex.
    # do this so can find vectors orthogonal to it
    for i in range(np.size(eigVecs,0)):
        orthoSys[i,:] = orthoSys[i,:]-eigVecs[mapVertToMicro[0],:]

    # find remaining vertices with Gram-Schmidt orthogonalization
    for k in range(1,nClusters):
        maxDist = 0

        # get previous vector of orthogonal basis set
        temp = orthoSys[mapVertToMicro[k-1],:].copy()

        # find vector in orthoSys that is most different from temp
        for i in range(np.size(eigVecs,0)):
            # remove basis vector just found (temp) so can find next orthogonal one
            orthoSys[i,:] = orthoSys[i,:]-np.dot(np.dot(orthoSys[i,:], np.transpose(temp)),temp)
            dist = np.linalg.norm(orthoSys[i,:])
            if dist > maxDist:
                maxDist = dist
                mapVertToMicro[k] = i

        orthoSys = orthoSys/maxDist

    return mapVertToMicro


def PCCA_Simplex(T,nMacro,doMinimization=True):
    P=PCCAPlusSolver(T)
    return P.PCCA_Simplex(nMacro-1,doMinimization=doMinimization)

class PCCAPlusSolver():

    def __init__(self,T):
        self.T=T
        self.nStates=T.shape[0]

    def PCCA_Simplex(self, nEigPerron, doMinimization=True):
        """Do PCCA clustering using the simplex method.  Should only be called by PCCA().
        
        ARGUMENTS:
        nEigPerron = number of eigenvalues close to 1 (int)
        doMinimization = whether or not to do minimization (bool)
        """
        from numpy import *
        from numpy.linalg import norm,inv
        # just want right eigenvectors corresponding to Perron clusters
        #(eigVals, eigVecs) = self.microMSM.getRightEigSolution(nEigPerron+1)
        eigVals,eigVecsLeft=MSMLib.GetEigenvectors(self.T,nEigPerron+1)
        eigVals,eigVecs=MSMLib.GetEigenvectors_Right(self.T,nEigPerron+1)

        print("Done Getting Eigenvalues")
        # get invariant density
        pi=eigVecsLeft[:,0]

        # number eigenvectors
        nEigVecs = shape(eigVecs)[1]

        # pi orthogonalization
        for i in range(nEigVecs):
            denom = dot(transpose(eigVecs[:,nEigVecs-1]*pi), eigVecs[:,nEigVecs-1])
            denom = sqrt(denom)
            denom *= sign(eigVecs[0,nEigVecs-1])
            eigVecs[:,nEigVecs-1] = eigVecs[:,nEigVecs-1] / denom

        maxScale = 0.0
        maxInd=0
        for i in range(nEigVecs):
            scale = sum(pi * eigVecs[:,i])
            if abs(scale) > maxScale:
                maxScale = abs(scale)
                maxInd = i
        eigVecs[:,maxInd] = eigVecs[:,0]
        eigVecs[:,0] = 1
        eigVecs[where(pi<=0)[0],:] = 0

        for i in range(1,nEigVecs):
            for j in range(i-1):
                scale = dot(transpose(eigVecs[:,j]*pi), eigVecs[:,i])
                eigVecs[:,i] -= scale*eigVecs[:,j]
            sumVal = sqrt(dot(transpose(eigVecs[:,i]*pi), eigVecs[:,i]))
            eigVecs[:,i] /= sumVal

        # find representative microstate for each vertex of simplex (cluster)
        repMicroState = self.findSimplexVertices(nEigPerron+1, eigVecs)

        # get initial guess for transformatin matrix A
        A = eigVecs[repMicroState,:]
        A = inv(A)
        normA = norm(A[1:nEigVecs, 1:nEigVecs])

        # check what min of chi is before optimize
        minChi = (dot(eigVecs,A)).min()
        print " Before optimize, chi.min = %f" % minChi

        # get flattened representation of A
        alpha = zeros([(nEigVecs-1)*(nEigVecs-1)])
        for i in range(nEigVecs-1):
            for j in range(nEigVecs-1):
                alpha[j + i*(nEigVecs-1)] = A[i+1,j+1]

        # do optimization
        from scipy.optimize import fmin
        initVal = -self.objectiveFunc(alpha, eigVecs, pi, normA)
        print " Initial value of objective function: %f" % initVal
        if doMinimization:
            alpha = fmin(self.objectiveFunc, alpha, args=(eigVecs, pi, normA), maxiter=1e6, maxfun=1e6)
        else:
            print " Skipping Minimization Step"

        # get A back from alpha
        for i in range(nEigVecs-1):
            for j in range(nEigVecs-1):
                A[i+1,j+1] = alpha[j + i*(nEigVecs-1)]

        # fill in missing values in A
        A[1:nEigVecs,0] = -sum(A[1:nEigVecs,1:nEigVecs], 1)
        for j in range(nEigVecs):
            A[0,j] = -dot(eigVecs[0,1:nEigVecs], A[1:nEigVecs,j])
            for l in range(1,self.nStates):
                dummy = -dot(eigVecs[l,1:nEigVecs], A[1:nEigVecs,j])
                if dummy > A[0,j]:
                    A[0,j] = dummy
        A /= sum(A[0,:])

        # find chi matrix, membership matrix giving something like probability that each microstate belongs to each vertex/macrostate.
        # say like probability because may get negative numbers and don't necessarily sum to 1 due imperfect simplex structure.
        # rows are microstates, columns are macrostates
        self.chi = dot(eigVecs, A)

        # print final values of things
        minChi = (dot(eigVecs,A)).min()
        print " At end, chi.min = %f" % minChi
        finalVal = -self.objectiveFunc(alpha, eigVecs, pi, normA)
        print " Final value of objective function: %f" % finalVal

        # find most probable mapping of all microstates to macrostates.
        self.mapMicroToMacro = argmax(self.chi,1)
        return self.mapMicroToMacro

    def objectiveFunc(self, alpha, eigVecs, pi, NORMA):
        from numpy import *
        from numpy.linalg import norm,inv
        nEigVecs = shape(eigVecs)[1]

        # get A back from alpha
        A = zeros([nEigVecs,nEigVecs])
        for i in range(nEigVecs-1):
            for j in range(nEigVecs-1):
                A[i+1,j+1] = alpha[j + i*(nEigVecs-1)]

        normA = norm(A[1:nEigVecs, 1:nEigVecs])

        # fill in missing values in A
        A[1:nEigVecs,0] = -sum(A[1:nEigVecs,1:nEigVecs], 1)
        for j in range(nEigVecs):
            A[0,j] = -dot(eigVecs[0,1:nEigVecs], A[1:nEigVecs,j])
            for l in range(1,self.nStates):
                dummy = -dot(eigVecs[l,1:nEigVecs], A[1:nEigVecs,j])
                if dummy > A[0,j]:
                    A[0,j] = dummy
        A /= sum(A[0,:])

        # optimizing trace(S)
        optval = trace(dot(dot(diag(1/A[0,:]), transpose(A)), A))
        optval = -(optval - (NORMA-normA)*(NORMA-normA))

        return optval
    
    def findSimplexVertices(self, nClusters, eigVecs):
        """Find the vertices of the simplex structure.  Do this by finding vectors that are as close as possible to orthogonal.

        ARGUMENTS:
          nClusters = number of Perron clusters (int)
          eigVecs = first nCluster eigenvectors (matrix of floats).  That is, eigenvectors corresponding to Perron clusters

        RETURN: list mapping between simplex vertices and represnetative microstates (microstate that lies exactly on vertex) (array of ints).
        """

        from numpy import *
        from numpy.linalg import norm,inv
        # initialize mapping between simplex verices and microstates
        mapVertToMicro = zeros([nClusters], int32)

        # copy of eigVecs that will use/modify to find orthogonal vectors
        orthoSys = eigVecs.copy()

        # find the first vertex, the eigenvector with the greatest norm (or length).
        # this will be the first of our basis vectors
        maxNorm = 0
        for i in range(size(eigVecs,0)):
            dist = norm(eigVecs[i,:])
            if dist > maxNorm:
                maxNorm = dist
                mapVertToMicro[0] = i

        # reduce every row of orthoSys by eigenvector of first vertex.
        # do this so can find vectors orthogonal to it
        for i in range(size(eigVecs,0)):
            orthoSys[i,:] = orthoSys[i,:]-eigVecs[mapVertToMicro[0],:]
          
        # find remaining vertices with Gram-Schmidt orthogonalization
        for k in range(1,nClusters):
            maxDist = 0

            # get previous vector of orthogonal basis set
            temp = orthoSys[mapVertToMicro[k-1],:].copy()

            # find vector in orthoSys that is most different from temp
            for i in range(size(eigVecs,0)):
                # remove basis vector just found (temp) so can find next orthogonal one
                orthoSys[i,:] = orthoSys[i,:]-dot(dot(orthoSys[i,:], transpose(temp)),temp)
                dist = norm(orthoSys[i,:])
                if dist > maxDist:
                    maxDist = dist
                    mapVertToMicro[k] = i
                  
            orthoSys = orthoSys/maxDist
        
        return mapVertToMicro
