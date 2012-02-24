import numpy as np
import scipy, scipy.linalg,scipy.optimize
import OldCorrelation

def SingleLikelihood(C,T):
    """Calculate the likelihood of a transition matrix given the observed counts C."""
    f=np.sum(C*np.log(T))
    return f

def ConvertTIntoK(T0):
    """Convert a transition matrix into a rate matrix.

    Inputs:
    T0: a transition matrix.

    Notes:
    For best results, use a transition matrix constructed with a short lagtime.
    This follows because one can show that
    T = exp(K t) ~ I +K t + ...
    """
    D=T0.diagonal()
    T=T0-np.diag(D)
    D2=T.sum(1)
    T=T-np.diag(D2)
    return(T)

def WeightedLikelihood(K,Ass,WhichTimes,Ckij,Weights=None):
    """Calculate the total likelihood of a rate matrix given assignment data.

    Inputs:
    K: rate array
    Ass: Assignments array
    WhichTimes: a set of lagtimes to use in likelihood calculation.
    Ckij: the array of count matrices Cij at each lagtime k.
    Optional Arguments
    Weights=None: a set of weights to weight the likelihoods at each lagtime.
    If none, weights are set to be uniform.
    """
    
    if Weights==None:
        Weights=ones(len(WhichTimes))
    NumStates=Ass.max()+1
    f=0.
    for k,Time in enumerate(WhichTimes):
        T=scipy.linalg.matfuncs.expm(K*Time)
        f+=np.sum(Ckij[Time]*np.log(T))*Weights[k]
    return f

def ConstructRateFromParams(X,Map,Pi):
    """Construct a rate matrix from a flat array of parameters.

    Inputs:
    X: flat array of parameters.
    Map: Mapping from flat indices to (2d) array indices.
    Pi: stationary populations of model

    """
    K=np.zeros((Pi.shape[0],Pi.shape[0]))
    K[Map.T[0],Map.T[1]]=abs(X)
    X2=abs(X)*Pi[Map.T[0]]/Pi[Map.T[1]]
    K[Map.T[1],Map.T[0]]=X2
    K-=np.diag(K.sum(1))
    return K

def GetParamsFromRate(Rate,Map):
    """Convert a rate array into a flat vector of parameters.

    Inputs:
    Rate: a rate array.
    Map: a Mapping from 1D indices to 2D indices.
    """

    X=Rate[Map.T[0],Map.T[1]]
    return X

def GetParamMapping(K):
    """Get a mapping from 1D to 2D indices.

    Inputs:
    K: a rate array
    """

    M=[]
    X=[]
    NumStates=K.shape[0]
    for i in range(NumStates):
        for j in range(NumStates):
            if i < j and K[i,j]>0:
                M.append([i,j])
                X.append(K[i,j])

    M=np.array(M)
    X=np.array(X)
    return M,X

def OptimizeRates(K0,Pi,Ass,WhichTimes,Ckij,Weights):
    """Maximize the likelihood of a rate matrix given assignment data.

    Inputs:
    K0: initial value of rate matrix.
    Pi: equilibrium populations.
    Ass: Assignments array
    WhichTimes: array of lagtimes to use in likelihood calculation.
    Ckij: array of count matrics Cij at each lagtime k
    Weights: weights to apply to data at each lagtime.
    """

    M,X=GetParamMapping(K0)

    def obj(X):
        K=ConstructRateFromParams(X,M,Pi)
        f=WeightedLikelihood(K,Ass,WhichTimes,Ckij,Weights=Weights)
        return -1*f

    def callback(X):
        print(obj(X))
    ans=scipy.optimize.fmin(obj,X,full_output=True,xtol=1E-10,ftol=1E-10,maxfun=100000,maxiter=100000,callback=callback)
    return ans
