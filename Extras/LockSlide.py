import pymc
import numpy as np
import scipy, scipy.linalg,scipy.optimize,scipy.stats
import OldCorrelation
import matplotlib
from msmbuilder import MSMLib

def FixEntry(M,X,Pi,K0,i,j,Val):
    """Constrain an entry in a rate matrix.

    Notes:
    Also constrains the transpose entry by detailed balance.
    Removes the entry from the lists of free variables (M, X).
    """
    Ind=M.index([i,j])
    M.pop(Ind)
    X.pop(Ind)
    K0[i,j]=Val
    K0[j,i]=Val*Pi[i]/Pi[j]

def LogLikelihood(C,T):
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

def ConstructRateFromParams(X,Map,Pi,K0):
    """Construct a rate matrix from a flat array of parameters.

    Inputs:
    X: flat array of parameters.
    Map: Mapping from flat indices to (2d) array indices.
    Pi: stationary populations of model
    K0: the baseline values

    """
    X=np.array(X)
    Map=np.array(Map)
    
    K=K0.copy()
    if len(Map)>0:
        K[Map.T[0],Map.T[1]]=abs(X)
        X2=abs(X)*Pi[Map.T[0]]/Pi[Map.T[1]]
        K[Map.T[1],Map.T[0]]=X2
    K-=np.diag(K.sum(1))
    return K

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
            if i > j and K[i,j]>0:
                M.append([i,j])
                X.append(K[i,j])

    return M, X
                

def MaximizeRateLikelihood(X,M,Pi,C,K0):
    """Maximize the likelihood of a rate matrix given assignment data.

    Inputs:
    K0: initial value of rate matrix.
    Pi: equilibrium populations.
    WhichTimes: array of lagtimes to use in likelihood calculation.
    Ckij: array of count matrics Cij at each lagtime k
    Weights: weights to apply to data at each lagtime.
    """

    def obj(X):
        K=ConstructRateFromParams(X,M,Pi,K0)
        T=scipy.linalg.matfuncs.expm(K)
        f=LogLikelihood(C,T)
        return -1*f

    def callback(X):
        pass
#        print("%.12f"%obj(X))
    ans=scipy.optimize.fmin(obj,X,full_output=True,xtol=1E-10,ftol=1E-10,maxfun=100000,maxiter=100000,callback=callback)[0]
    ans=abs(ans)
    return ans

def PlotRates(KList,LagTimeList,Tau=1,CountsList=None):
    KList=np.array(KList)
    if CountsList!=None:
        CountsList=np.array(CountsList)
    NumStates=KList.shape[-1]
    TauList=Tau/KList
    for i in range(NumStates):
        for j in range(NumStates):
            if i > j and KList[0,i,j]> 0:
                if CountsList==None:
                    matplotlib.pyplot.plot(Tau*LagTimeList,TauList[:,i,j],label="%d-%d"%(i,j))
                else:
                    matplotlib.pyplot.errorbar(Tau*LagTimeList,TauList[:,i,j],yerr=TauList[:,i,j]/np.sqrt(CountsList[:,i]),label="%d-%d"%(i,j))

    matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.legend(loc=0)

class AdaptiveRateEstimator():
    """
    Notes:

    """
    def __init__(self,Ass,MinLagTime=1,MaxLagTime=np.inf,Cutoff=0.45):
        self.Ass=Ass
        self.Lagtime=MinLagTime
        self.MaxLagTime=MaxLagTime
        self.Cutoff=Cutoff

        C0=MSMLib.GetCountMatrixFromAssignments(self.Ass,LagTime=MinLagTime)
        C0=MSMLib.ErgodicTrim(C0,self.Ass)[0]
        C0=MSMLib.IterativeDetailedBalance(C0)
        T0=MSMLib.EstimateTransitionMatrix(C0).toarray()
        self.NumStates=Ass.max()+1
        self.K=ConvertTIntoK(T0)
        self.Pi=np.array(C0.sum(0)).flatten()
        self.Pi/=self.Pi.sum()

        self.M,self.X=GetParamMapping(self.K)
        self.LagTime=1

    def Estimate(self):
        while len(self.M)>0:
            print("Estimating rates with lagtime %d"%self.LagTime)
            self.Iterate()

        self.K=ConstructRateFromParams(self.X,self.M,self.Pi,self.K)

    def Iterate(self):
        K1=self.K.copy()
        K2=self.K.copy()

        LagTime1=self.LagTime
        LagTime2=self.LagTime+1

        C1=MSMLib.GetCountMatrixFromAssignments(self.Ass,LagTime=LagTime1).toarray()
        C1/=float(LagTime1)
        X1=MaximizeRateLikelihood(self.X,self.M,self.Pi,C1,self.K)
        K1=ConstructRateFromParams(X1,self.M,self.Pi,self.K)
        K1/=(LagTime1)

        C2=MSMLib.GetCountMatrixFromAssignments(self.Ass,LagTime=LagTime2).toarray()
        C2/=float(LagTime2)
        X2=MaximizeRateLikelihood(self.X,self.M,self.Pi,C2,self.K)
        K2=ConstructRateFromParams(X2,self.M,self.Pi,self.K)
        K2/=(LagTime2)

        self.CompareRatesMCMC(K1,K2,C1,C2,X1,X2,LagTime1,LagTime2)
        self.LagTime=np.ceil(self.LagTime*1.25)

    def CompareRatesMCMC(self,K1,K2,C1,C2,X1,X2,LagTime1,LagTime2,MaxRate=10.,NumIter=250000,burn=1000,thin=25,ZCutoff=0.13):
        N=len(X1)

        print(X1)
        print(X2)
        U1 = pymc.Uniform('U1', np.zeros(N),MaxRate*np.ones(N),value=X1)
        U2 = pymc.Uniform('U2', np.zeros(N),MaxRate*np.ones(N),value=X2)

        @pymc.potential
        def f1(U1=U1):
            K=ConstructRateFromParams(U1,self.M,self.Pi,self.K)
            T=scipy.linalg.matfuncs.expm(K*LagTime1)
            L=LogLikelihood(C1,T)
            return L

        @pymc.potential
        def f2(U2=U2):
            K=ConstructRateFromParams(U2,self.M,self.Pi,self.K)
            T=scipy.linalg.matfuncs.expm(K*LagTime2)
            L=LogLikelihood(C2,T)
            return L
        
        MC1 = pymc.MCMC([U1, f1])
        MC2 = pymc.MCMC([U2, f2])

        MC1.sample(NumIter,burn=burn,thin=thin)
        MC2.sample(NumIter,burn=burn,thin=thin)

        stat1=U1.stats()
        stat2=U2.stats()

        t1=U1.trace()
        t2=U2.trace()
        d=t2-t1
        N=len(t1)

        mu=d.mean(0)
        sig=d.std(0)
        z=abs(mu/sig)

        print("tau")
        print(1./stat1["mean"])
        print("mean")
        print(mu)
        print("std")
        print(sig)
        print("z")
        print(z)

        RemovalList=[]
        for i in range(len(self.M)):
            a,b=self.M[i]
            factor=1./(1+self.Pi[a]/self.Pi[b])
            CurrentTau=factor/stat1["mean"]
            print("CurrentTau = ",CurrentTau)
            if np.rank(CurrentTau)>0:
                CurrentTau=CurrentTau[i]
            if z[i] < ZCutoff or (CurrentTau <= LagTime1):
                RemovalList.append(self.M[i])

        for [i,j] in RemovalList:
            print("Fixing %d-%d"%(i,j))
            LockSlide.FixEntry(self.M,self.X,self.Pi,self.K,i,j,K1[i,j])
