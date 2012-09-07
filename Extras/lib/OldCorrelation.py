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

import numpy as np
from msmbuilder import MSMLib
    
def Correlation(x,Normalize=True,MaxTime=np.inf):
    """Calculate the (auto) correlation function of an sequence of observables x.
    """
    x=x-np.mean(x)
    n=x.shape[0]
    ACF=[np.var(x)]
    for t in xrange(1,min(MaxTime,n)):
        ACF.append(np.dot(x[t:],x[:-t])/(float(n)-float(t)))
    ACF=np.array(ACF)
    if Normalize==True:
        ACF/=ACF[0]
    return(ACF)

def RawMSMCorrelation(T,ObservableArray,AssignmentArray,Steps=10000,StartingState=0,Normalize=True):
    """Calculate an autocorrelation function from an MSM.  Inputs include: Transition Matrix T, an array of the observable calculated at every project frame, and the array of assignments.  This function works by first generating a 'sample' trajectory from the MSM.  This approach is necessary as it allows treatment of intra-state dynamics.
    """
    Traj=MSMLib.Sample(T,StartingState,Steps)
    ObsTraj=[]
    for k,State in enumerate(Traj):
        Obs=ObservableArray[np.where(AssignmentArray==State)]
        MaxNum=len(Obs)
        Samp=np.random.random_integers(0,MaxNum-1)
        ObsTraj.append(Obs[Samp])
    ObsTraj=np.array(ObsTraj)
    Cor=Correlation(ObsTraj,Normalize=Normalize)
    return(Cor,Traj,ObsTraj)


def GetVectorCorrelation(v,Ass,NMax=None):
    NTraj=Ass.shape[0]
    if NMax==None:
        NMax=Ass.shape[1]
    y=np.zeros(NMax)
    for j in range(NTraj):
        trj=v[Ass[j]]
        y+=Correlation(trj,False,NMax)
    y/=float(NTraj)
    return y


def ReOrderEigensystem(Ass,Lam,EV,Factor=20.,Cutoff=1./np.e,CorrelationLength=50):
    """Fix ordering issues in eigenvalues.  This is a helper function for GetCorrelationCorrectedEigensystem.

    Inputs:

    Ass: Assignments array
    Lam: Eigenvalue array
    EV: RIGHT eigenvector array

    Eigenvalues  estimated using a short lagtime model generally yield timescales that are far too fast.  In particular, often those timescales would be slower when estimated with a longer timescale model.  This is particularly bad when using PCCA or PCCA+, as sometimes the short-timescale eigenvalues are not in correct rank order.  This leads to poor state decompositions.  This functino uses an eigenvector correlation function analysis to get the 'long lagtime' corrected eigenvalues, in the correct order.  These eigenvalues can then be used to yield a better state decomposition.
    """
    
    Tau=-1./np.log(Lam)
    RealTau=Tau.copy()

    NumEigen=len(Lam)
    for i in range(1,NumEigen):
	y=GetVectorCorrelation(EV[:,i],Ass,CorrelationLength)
	y/=y[0]
	x=1.*np.arange(len(y))
        try:
            RealTau[i]=x[np.where(y<Cutoff)][0]
        except:
            RealTau[i]=-len(x)/np.log(y[-1])

    Ind=np.argsort(abs(RealTau))[::-1]
    Lam2=np.exp(-1./RealTau)
    return Ind,Lam2

def GetCorrelationCorrectedEigensystem(T,NumEigen,Assignments,MultiplicativeFactor=6,CorrelationLength=50):
    """Get the slowest eigenvalues of a system, correcting for nonmarkovian bias.

    Inputs:
    T: Transition Matrix
    NumEigen: Number of eigenvalues to get
    Assignments: Assignments array

    Optional Arguments:
    

    Eigenvalues  estimated using a short lagtime model generally yield timescales that are far too fast.  In particular, often those timescales would be slower when estimated with a longer timescale model.  This is particularly bad when using PCCA or PCCA+, as sometimes the short-timescale eigenvalues are not in correct rank order.  This leads to poor state decompositions.  This functino uses an eigenvector correlation function analysis to get the 'long lagtime' corrected eigenvalues, in the correct order.  These eigenvalues can then be used to yield a better state decomposition.
    """

    #We first calculate more than the desired number of eigenvalues
    #Then we correct them and pick the slowest *corrected* eigenvalues
    NumEigenToCalculate=NumEigen*MultiplicativeFactor
    
    eigVals,eigVecs=MSMLib.GetEigenvectors(T,NumEigenToCalculate)
    #eigVals,eigVecs_Right=MSMLib.GetEigenvectors_Right(T,(NumEigen)*MultiplicativeFactor)

    #Calculate the right eigenvectors using the stationary vector
    eigVecs_Right=eigVecs.copy()
    Pi=eigVecs[:,0]
    for i in range(NumEigenToCalculate):
        eigVecs_Right[:,i]/=Pi
    
    Ind,CorrelationEigVals=ReOrderEigensystem(Assignments,eigVals,eigVecs_Right,CorrelationLength=CorrelationLength)

    #Re-order using the correct ordering
    CorrelationEigVals=CorrelationEigVals[Ind]
    eigVals=eigVals[Ind]
    eigVecs=eigVecs[:,Ind]
    eigVecs_Right=eigVecs_Right[:,Ind]

    #Collect the NumEigen slowest eigenvalues and eigenvectors.
    eigVals=eigVals[0:NumEigen]
    eigVecs=eigVecs[:,0:NumEigen]
    eigVecs_Right=eigVecs_Right[:,0:NumEigen]
    CorrelationEigVals=CorrelationEigVals[0:NumEigen]
    
    print(-1/np.log(eigVals))                  
    print(-1/np.log(CorrelationEigVals))

    return eigVals,CorrelationEigVals,eigVecs, eigVecs_Right
