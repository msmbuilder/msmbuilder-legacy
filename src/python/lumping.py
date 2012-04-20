import scipy.optimize
import scipy.io
import numpy as np

from msmbuilder import MSMLib

from numpy import dot, diag
inv=np.linalg.inv
norm = np.linalg.norm
tr = np.trace

def NormalizeLeftEigenvectors(V):
    """Normalize the left eigenvectors, such that <v[:,i]/pi, v[:,i]> = 1

    Notes:

    Acts inplace.

    Assumes that V[:,0] is the equilibrium vector and that detailed balance holds.
    """
    pi=V[:,0]
    pi/=pi.sum()
    
    for k in xrange(1,V.shape[-1]):
        x=V[:,k]
	x/=abs(np.dot(x/pi,x))**.5

def trim_eigenvectors_by_flux(lam, vl, flux_cutoff):
    """Trim eigenvectors that have low equilibrium flux.

    Notes:

    Assuming that the left eigenvectors are properly pi-normalized,
    the equilibrium flux contributino of each eigenvector v is given by

    \sum_i v_i^2
    
    """
    NormalizeLeftEigenvectors(vl)

    N = len(lam)

    flux_list = np.array([(vl[:,i]**2).sum() for i in range(N)])
    flux_list /= flux_list.max()
    flux_list[0] = 1.
    KeepInd = np.where(flux_list>=flux_cutoff)[0]

    print("Implied timescales (UNITLESS)")
    print(-1/np.log(lam))
    print("Flux")
    print(flux_list)
    print("Keeping %d eigenvectors after flux cutoff %f"%(len(KeepInd),flux_cutoff))

    lam = lam[KeepInd]
    vl = vl[:,KeepInd]
    flux_list = flux_list[KeepInd]
    
    print("After Flux calculation, Implied timescales (UNITLESS):")
    print(-1/np.log(lam))

    print("After Flux calculation, fluxes.")
    print(flux_list)

    return lam, vl

def get_maps(A):
    """Helper function for PCCA+ optimization.  Get mappings from the square array A to the flat vector of parameters alpha.
    """

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

def to_flat(A, flat_map):
    """Convert a square matrix A to a flat array alpha."""
    return A[flat_map[:,0],flat_map[:,1]]

def to_square(alpha, square_map):
    """Convert a flat array alpha to a square array A."""
    return alpha[square_map]

def pcca_plus(T, N, flux_cutoff=None, do_minimization=True, min_population=0.0,objective_function = "crisp_metastability"):
    """Perform PCCA+.

    Inputs:
    T -- transition matrix, csr format.
    N -- desired (maximum) number of macrostates.
    """
    n = T.shape[0]
    lam,vl = MSMLib.GetEigenvectors(T,N)
    NormalizeLeftEigenvectors(vl)

    if flux_cutoff != None:
        lam,vl = trim_eigenvectors_by_flux(lam,vl, flux_cutoff)
        N = len(lam)
    
    pi = vl[:,0]

    vr = vl.copy()
    for i in range(N):
        vr[:,i] /= pi
        vr[:,i] *= np.sign(vr[0,i])
        vr[:,i] /= np.sqrt(dot(vr[:,i]*pi,vr[:,i]))

    A, chi, microstate_mapping = opt_soft(vr, N, pi, lam, T, do_minimization=do_minimization, min_population=min_population,objective_function=objective_function)

    return A, chi,vr, microstate_mapping


def opt_soft(vr, N, pi, lam, T, do_minimization=True, use_anneal=True, min_population=0.0,objective_function="crisp_metastability"):
    """Core routine for PCCA+ algorithm.
    """
    n = len(vr[0])

    index = index_search(vr)

    # compute transformation matrix A as initial guess for local optimization (maybe not feasible) 
    A = vr[index,:]

    A = inv(A)
    A = fill_A(A, vr )

    if do_minimization==True:
        flat_map, square_map = get_maps(A)
        alpha = to_flat(1.0*A,flat_map)

        obj = lambda x: -1*objective(x,vr,square_map,lam,T,pi,objective_function=objective_function,min_population=min_population)

        print("Initial value of objective function: %f"%obj(alpha))

        alpha = scipy.optimize.anneal(obj,alpha,lower=0.0,maxiter=1,schedule="boltzmann",dwell=1000,feps=1E-3,boltzmann=2.0,T0=1.0)[0]

        alpha = scipy.optimize.fmin(obj,alpha,full_output=True,xtol=1E-4,ftol=1E-4,maxfun=5000,maxiter=100000)[0]

        print("*********")
        print("Final values.\n f = %f"%(-1*obj(alpha)))

        A = to_square(alpha,square_map)

    else:
        print("Skipping Minimization")

    A = fill_A(A, vr )

    chi = dot(vr,A)

    microstate_mapping = np.argmax(chi,1)

    return A, chi, microstate_mapping

def has_constraint_violation(A,vr,epsilon = 1E-8):
    """Check for constraint violations using Eqn 4.25."""

    lhs = 1-A[0,1:].sum()
    rhs = rhs= dot(vr[:,1:],A[1:,0])
    rhs = -1*rhs.min()
    
    if abs(lhs-rhs) > epsilon:
        return True

def objective(alpha,vr,square_map,lam,T,pi,barrier_penalty=20000.,objective_function="crisp_metastability",min_population=0.):
    """Return the PCCA+ objective function.

    Notes: three choices of objective_function:

    crispness: try to make crisp state decompostion (recommended)

    metastability: try to make metastable fuzzy state decomposition

    crisp_metastability: try to make the resulting crisp msm metastable

    """
    n,N=vr.shape

    A = to_square(alpha, square_map)

    #make A feasible
    A = fill_A(A, vr)

    chi_fuzzy = dot(vr,A)
    mapping = np.argmax(chi_fuzzy,1)

    possible_objective_functions = ["crispness","crisp_metastability","metastability"]
    
    if objective_function not in possible_objective_functions:
        raise Exception("objective_function must be one of ",possible_objective_functions)
    
    if objective_function == "crisp_metastability":
        chi = 0.0*chi_fuzzy
        chi[np.arange(n),mapping] = 1.
    elif objective_function == "metastability":
        chi = chi_fuzzy

    if objective_function in ["crisp_metastability","metastability"]:
        #Calculate  metastabilty of the lumped model.  Eqn 4.20 in LAA.
        meta = 0.
        for i in range(N):
            meta += dot(T.dot(chi[:,i]),pi*chi[:,i]) / dot(chi[:,i],pi)
        
        obj = meta

    if objective_function == "crispness":
        #Calculate the crispness defined by Roeblitz Doctoral thesis.
        obj = tr(dot(diag(1./A[0]),dot(A.transpose(),A)))

    """
    If microstate_mapping is degenerate (not enough macrostates), we increase obj
    This prevents PCCA+ returning a model with insufficiently many states.
    We also do this if the macrostate populations are too low.
    """

    pi_macro = np.array([pi[mapping==i].sum() for i in range(N)])
    if len(np.unique(mapping))!= N or has_constraint_violation(A,vr) or pi_macro.min() < min_population:
        print("Warning: constraint violation detected.")
        obj -= barrier_penalty

    print("f = %f"%(obj))

    return obj

def fill_A(A,vr):
    """Make A feasible.  
    """
    
    n,N=vr.shape
    
    A = A.copy()
    
    #compute 1st column of A by row sum condition 
    A[1:,0] = -1*A[1:,1:].sum(1)

    # compute 1st row of A by maximum condition
    A[0] = -1*dot(vr[:,1:],A[1:]).min(0)

    """Obsolete code replaced by previous line
    for j in range(N):

        A[0,j] = -1*dot(vr[0,1:],A[1:,j])
        
        for l in range(1,n):
            dummy = -1* dot(vr[l,1:],A[1:,j])
            if dummy > A[0,j]:
                A[0,j] = dummy
    """

    #rescale A to be in the feasible set
    A /= A[0].sum()

    return A


def index_search(vr):
    """Find a simplex structure in the data.  First step in PCCA+ algorithm.
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

def PCCA(T,num_macro,tolerance=1E-5,flux_cutoff=None):
    """Create a lumped model using the PCCA algorithm.  

    Inputs:
    T -- A transition matrix.  
    num_macro -- The desired number of states.

    Optional Inputs:
    tolerance=1E-5 : specifies the numerical cutoff to use when splitting states based on sign.

    Returns a mapping from the Microstate indices to the Macrostate indices.
    To construct a Macrostate MSM, you then need to map your Assignment data to the new states (e.g. Assignments=MAP[Assignments]).
    """

    n = T.shape[0]
    lam,vl = MSMLib.GetEigenvectors(T,num_macro)
    NormalizeLeftEigenvectors(vl)

    if flux_cutoff != None:
        lam,vl = trim_eigenvectors_by_flux(lam,vl, flux_cutoff)
        num_macro = len(lam)
    
    pi = vl[:,0]

    vr = vl.copy()
    for i in range(num_macro):
        vr[:,i] /= pi
        vr[:,i] *= np.sign(vr[0,i])
        vr[:,i] /= np.sqrt(dot(vr[:,i]*pi,vr[:,i]))

    #Remove the stationary eigenvalue and eigenvector.
    lam = lam[1:]
    vr = vr[:,1:]

    microstate_mapping = np.zeros(n,'int')

    #Function to calculate the spread of a single eigenvector.
    spread = lambda x: x.max()-x.min()
    """
    1.  Iterate over the eigenvectors, starting with the slowest.
    2.  Calculate the spread of that eigenvector within each existing macrostate.
    3.  Pick the macrostate with the largest eigenvector spread.
    4.  Split the macrostate based on the sign of the eigenvector.  
    """
    
    for i in range(num_macro-1):#Thus, if we want 2 states, we split once.
        v = vr[:,i]
        AllSpreads = np.array([spread(v[microstate_mapping==k]) for k in range(i+1)])
        StateToBeSplit = np.argmax(AllSpreads)
        microstate_mapping[(microstate_mapping==StateToBeSplit)&(v>=tolerance)] = i+1

    return(microstate_mapping)
