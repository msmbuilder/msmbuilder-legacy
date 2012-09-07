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

"""Tools for working with instantaneous rate matrices (e.g. master equations).
"""

def GetRateMatrix(T,EigAns=None,FixNegativity=True):
    NumStates=T.shape[0]
    if EigAns==None:
        EigAns=MSMLib.GetEigenvectors(T,NumStates)
    Pi=EigAns[1][:,0]
    print("Done Getting Eigenvectors")
    """
    K=np.zeros((NumStates,NumStates),dtype=T.dtype)
    for i in range(1,NumStates):
	phi=EigAns[1][:,i]
	psi=phi/Pi
	alpha=np.dot(phi,psi)**.5
	psi/=alpha
	phi/=alpha
	K-=np.log(EigAns[0][i])*np.outer(psi,phi)
    """
    #To Check, compare the following transition matrix with the input:
    #T2=scipy.linalg.matfuncs.expm(-K)
    #T2-T

    ev=EigAns[1]
    p=ev[:,0]
    for i in xrange(NumStates):
	ev[:,i]/=np.dot(ev[:,i]/p,ev[:,i])**.5

    #Ld=np.diag(-np.log(EigAns[0]))
    #K=np.dot(np.dot(np.dot(np.diag(1./Pi),ev),Ld),ev.transpose())
    #return(K)
    print("Getting evT and deleting old ev.")
    ev=np.real(ev).copy()

    lam=EigAns[0]
    lam=np.abs(lam)
    #lam[where(lam<0)]=1/np.e #Anything with negative eigenvalues is set to have timescale 1 lagtime.


    del EigAns


    D=scipy.sparse.dia_matrix((1/p,0),(NumStates,NumStates))
    K=D.dot(ev)
    print("Done 1st mm")

    L=scipy.sparse.dia_matrix((-np.log(lam),0),(NumStates,NumStates))
    K=K.transpose()
    K=L.dot(K)
    print("Done 2nd mm")
    K=K.transpose()
    K=np.array(K,dtype=lam.dtype,order="C")

    ev=ev.transpose()
    ev=np.array(ev,dtype=lam.dtype,order="C")
    K=np.dot(K,ev)
    print("Done 3rd mm")

    if FixNegativity==True:#This enforces "reasonable" constraints on the rate matrix, e.g. negative off diagonals and positive diagonals
        RemoveRateDiagonal(K)
    return(K)

def ReconstructTransitionMatrix(K,RateEigAns=None):
    NumStates=K.shape[0]
    if RateEigAns==None:
        RateEigAns=np.linalg.eig(K.transpose())
        Ind=np.argsort(RateEigAns[0])
        RateEigAns=(RateEigAns[0][Ind],RateEigAns[1][:,Ind])
        RateEigAns[1][:,0]/=RateEigAns[1][:,0].sum()

    Pi=RateEigAns[1][:,0]
    print("Done Getting Eigenvectors")
    T=np.zeros((NumStates,NumStates),dtype=K.dtype)
    for i in range(NumStates):
	phi=RateEigAns[1][:,i]
	psi=phi/Pi
	alpha=np.dot(phi,psi)**.5
	psi/=alpha
	phi/=alpha
        T+=np.exp(-RateEigAns[0][i])*np.outer(psi,phi)

    return(T,RateEigAns)

def GetDiagPerturbation(K,g,SymmetricPerturbation=False):
    """Here, g = Pi2 / Pi are the perturbed populations divided by the unperturbed populations."""
    NumStates=K.shape[0]
    if SymmetricPerturbation==True:
        g=g**0.5
    D=scipy.sparse.dia_matrix((g,0),(NumStates,NumStates))
    K2=D.dot(K.transpose())
    K2=K2.transpose()
    if SymmetricPerturbation==True:
        D=scipy.sparse.dia_matrix((1/g,0),(NumStates,NumStates))
        K2=D.dot(K2)
    RemoveRateDiagonal(K2)  
    return(K2)

def GetHammondPerturbation(K,Pi2,g,Anti=False):
    """Here, Pi2 are the perturbed populations."""
    NumStates=K.shape[0]

    K2=np.outer(Pi2,1/Pi2)
    if Anti==False:
        IND=np.where(K2<1)
    else:
        IND=np.where(K2>1)

    K2=np.outer(1/g,g)
    K2[IND]=1.
    K2*=K
    
    RemoveRateDiagonal(K2)  
    return(K2)

def RemoveRateDiagonal(K,FixNegativity=True):
    NumStates=K.shape[0]
    for i in xrange(NumStates):
        K[i,i]=0.

    if FixNegativity==True:#This enforces "reasonable" constraints on the rate matrix, e.g. negative off diagonals and positive diagonals
        for i in xrange(NumStates):
            K[i]-=np.abs(K[i])
        K*=0.5
    S=np.array(K.sum(1)).flatten()
    for i in xrange(NumStates):
        K[i,i]=-1*S[i]

def MatrixExp(K,NumIter=20):
    """Calculate the matrix exponential of K using power series expansion.  The scipy version uses too much memory, so I wrote this.  PS you should ensure that your rate matrix has correct sign structure by RemoveRateDiagonal() first."""

    T3=np.eye(K.shape[0],K.shape[0],dtype=K.dtype)
    T3+=K
    KTemp=K.copy()

    for i in range(2,NumIter):
	print(i)
	KTemp=np.dot(K,KTemp)
	KTemp/=float(i)
	T3+=KTemp

    return(T3)
