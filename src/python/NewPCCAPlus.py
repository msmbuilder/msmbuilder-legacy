import scipy.optimize
import scipy.io
import numpy as np

from Emsmbuilder import MSMLib

from numpy import dot, diag
inv=np.linalg.inv
norm = np.linalg.norm
tr = np.trace

def trim_eigenvectors_by_flux(lam, vl, flux_cutoff):
    """Trim eigenvectors that have low equilibrium flux.
    """
    MSMLib.NormalizeLeftEigenvectors(vl)

    N = len(lam)

    flux_list = np.array([(vl[:,i]**2).sum() for i in range(N)])
    flux_list /= flux_list[0]
    KeepInd = np.where(flux_list>=flux_cutoff)[0]

    KeepInd = [0,1,np.argmin(flux_list)]
    KeepInd = np.array(KeepInd)
    
    print("Implied timescales (UNITLESS)")
    print(-1/np.log(lam))
    print("Flux")
    print(flux_list)
    print("Keeping %d eigenvectors after flux cutoff %f"%(len(KeepInd),flux_cutoff))

    lam = lam[KeepInd]
    vl = vl[:,KeepInd]
    
    print("After Flux calculation, Implied timescales (UNITLESS):")
    print(-1/np.log(lam))

    return lam, vl

def get_maps(A):

    N = A.shape[0]
    flat_map = []
    for i in range(1,N):
        for j in range(1,N):
            flat_map.append([i,j])

    flat_map = np.array(flat_map)

    square_map = np.zeros(A.shape,'int')

    for k in range((N-1)**2):
        i,j = flat_map[k]
        square_map[i,j] = k

    return flat_map,square_map

def to_flat(A,flat_map):
    return A[flat_map[:,0],flat_map[:,1]]

def to_square(Alpha, square_map):
    return Alpha[square_map]

def pcca_plus(T,N, flux_cutoff=None,do_minimization=True):
    """Perform PCCA+.

    Inputs:
    T -- transition matrix, csr format.
    N -- desired (maximum) number of macrostates.
    """
    n = T.shape[0]
    lam,vl = MSMLib.GetEigenvectors(T,N)
    MSMLib.NormalizeLeftEigenvectors(vl)

    if flux_cutoff != None:
        lam,vl = trim_eigenvectors_by_flux(lam,vl, flux_cutoff)
        N = len(lam)
    
    pi = vl[:,0]

    vr = vl.copy()
    for i in range(N):
        vr[:,i] /= pi
        vr[:,i] *= np.sign(vr[0,i])
        vr[:,i] /= np.sqrt(dot(vr[:,i]*pi,vr[:,i]))

    A, chi, microstate_mapping = opt_soft(vr,N,pi,lam,do_minimization=do_minimization)

    return A, chi,vr, microstate_mapping


def opt_soft(vr,N,pi,lam,do_minimization=True):
    """
    """
    n = len(vr[0])

    index = index_search(vr)

    # compute transformation matrix A as initial guess for local optimization (maybe not feasible) 
    A = vr[index,:]

    A = inv(A)

    if do_minimization==True:
        flat_map, square_map = get_maps(A)
        alpha = to_flat(1.0*A,flat_map)

        obj = lambda x: objective(x,vr,square_map,lam)

        print("Initial value of objective function: %f"%obj(alpha))
    
        alpha = scipy.optimize.fmin(obj,alpha,full_output=True,xtol=1E-4,ftol=1E-4,maxfun=5000,maxiter=100000)[0]

        print("Final value of objective function: %f"%obj(alpha))

        A = to_square(alpha,square_map)

    else:
        print("Skipping Minimization")

    A = fill_A(A, vr )

    chi = dot(vr,A)

    microstate_mapping = np.argmax(chi,1)

    return A, chi, microstate_mapping

def objective(alpha,vr,square_map,lam):
    """
    """
    n,N=vr.shape

    A = to_square(alpha, square_map)

    #make A feasible
    A = fill_A(A, vr)

    chi = dot(vr,A)
    mapping = np.argmax(chi,1)

    """
    NOTE: IMHO the scaling objective function doesn't work.
    if which_objective == 'scaling':
        obj = -1*chi[np.arange(n),mapping].sum()
       #Calculated using equation 4.19 in LAA 2005 paper.
    """

    """Calculated using equation 4.22 in LAA 2005 paper."""
    X = dot(diag(lam),A**2.)
    X = dot(X, diag(1./A[0]))
    obj = -1*X.sum()

    #if microstate_mapping is degenerate (not enough macrostates), we set obj to infinity
    if len(np.unique(mapping))!= N:
        obj = np.inf
    
    print(-1*obj,"Det = ",np.linalg.det(A))

    return obj


def fill_A(A,vr):
    """Make A feasible.
    """
    
    n,N=vr.shape
    
    A = A.copy()
    
    #compute 1st column of A by row sum condition 
    A[1:,0] = -1*A[1:,1:].sum(1)

    # compute 1st row of A by maximum condition
    for j in range(N):

        A[0,j] = -1*dot(vr[0,1:],A[1:,j])
        
        for l in range(1,n):
            dummy = -1* dot(vr[l,1:],A[1:,j])
            if dummy > A[0,j]:
                A[0,j] = dummy

    A /= A[0].sum()#rescale A to be in the feasible set

    return A


def index_search(vr):
    """Find a simplex structure in the data.
    """

    n,N=vr.shape

    index=np.zeros(N,'int')
    
    # first vertex: row with largest norm
    index[0] = np.argmax( [norm(vr[i]) for i in range(n)])

    OrthoSys = vr - np.outer(np.ones(n),vr[index[0]])

    for j in range(1,N):
        temp = OrthoSys[index[j-1]].copy()
        for l in range(n):
            OrthoSys[l]-=temp*dot(OrthoSys[l],temp)

        distlist = np.array([norm(OrthoSys[l]) for l in range(n)])

        index[j] = np.argmax(distlist)

        OrthoSys /= distlist.max()

    return index



def iterative_pcca_plus(T,N,assignments0 ,population_cutoff=None,do_minimization=False):
    """Perform PCCA+.

    Inputs:
    T -- transition matrix, csr format.
    N -- desired (maximum) number of macrostates.
    """
    n = T.shape[0]
    lam,vl = MSMLib.GetEigenvectors(T,N)
    MSMLib.NormalizeLeftEigenvectors(vl)

    pi = vl[:,0]

    vr = vl.copy()
    for i in range(N):
        vr[:,i] /= pi
        vr[:,i] *= np.sign(vr[0,i])
        vr[:,i] /= np.sqrt(dot(vr[:,i]*pi,vr[:,i]))

    good_indices = [0]
    unknown_indices = range(1,N)
    bad_indices = []

    while True:
        if len(unknown_indices)==0:
            break
        print("good = ",good_indices)
        k = unknown_indices.pop(0)
        good_indices.append(k)

        vr1=vr[:,good_indices]
        lam1=lam[good_indices]
        N1=len(good_indices)
        A, chi, microstate_mapping = opt_soft(vr1,N1,pi,lam1,do_minimization=do_minimization)
        assignments=assignments0.copy()
        MSMLib.ApplyMappingToAssignments(assignments,microstate_mapping)
        
        CMacro = MSMLib.GetCountMatrixFromAssignments(assignments)
        CMacro, mapping2 = MSMLib.ErgodicTrim(CMacro)
        CMacro = MSMLib.IterativeDetailedBalance(CMacro)
        pi_macro = CMacro.sum(0)
        pi_macro /= pi_macro.sum()

        print(population_cutoff)
        print(pi_macro)
        
        if pi_macro.min() < population_cutoff:
            print("rejecting %d"%k)
            good_indices.pop()
            bad_indices.append(k)        

    vr1=vr[:,good_indices]
    lam1=lam[good_indices]
    N1=len(good_indices)
    A, chi, microstate_mapping = opt_soft(vr1,N1,pi,lam1,do_minimization=do_minimization)
    assignments=assignments0.copy()
    MSMLib.ApplyMappingToAssignments(assignments,microstate_mapping)
    
    CMacro = MSMLib.GetCountMatrixFromAssignments(assignments)
    CMacro, mapping2 = MSMLib.ErgodicTrim(CMacro)
    CMacro = MSMLib.IterativeDetailedBalance(CMacro)
    pi_macro = CMacro.sum(0)
    pi_macro /= pi_macro.sum()

    print(pi_macro)

    return A, chi,vr, microstate_mapping
