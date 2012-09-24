def GetPsiAtomIndicesAndPosition(C1):
    a=C1["AtomNames"]
    aC=np.where(a=="C")[0]
    aN=np.where(a=="N")[0]
    aCA=np.where(a=="CA")[0]
    X=C1["XYZ"]
    NumResi=C1.GetNumberOfResidues()
    return(NumResi,a,aC,aN,aCA,X)

def GetAllPsi(C1,NumResi=None,a=None,aC=None,aN=None,aCA=None,X=None):
    if a==None:
    	NumResi,a,aC,aN,aCA,X=GetPsiAtomIndicesAndPosition(C1)
    Data=[]
    for i in xrange(NumResi-1):
	print(i)
        x0=X[aN[i]]
        x1=X[aCA[i+1]]
        x2=X[aC[i+1]]
        x3=X[aN[i+1]]
        Psi=Torsion(x0,x1,x2,x3)
        Data.append(Psi)
    return(np.array(Data))
def GetPhiAtomIndicesAndPosition(C1):
    a=C1["AtomNames"]
    aC=np.where(a=="C")[0]
    aN=np.where(a=="N")[0]
    aCA=np.where(a=="CA")[0]
    X=C1["XYZ"]
    NumResi=C1.GetNumberOfResidues()
    return(NumResi,a,aC,aN,aCA,X)

def GetAllPhi(C1,NumResi=None,a=None,aC=None,aN=None,aCA=None,X=None):
    if a==None:
    	NumResi,a,aC,aN,aCA,X=GetPhiAtomIndicesAndPosition(C1)
    Data=[]
    for i in xrange(NumResi-1):
	print(i)
        x0=X[aC[i]]
        x1=X[aN[i+1]]
        x2=X[aCA[i+1]]
        x3=X[aC[i+1]]
        Phi=Torsion(x0,x1,x2,x3)
        Data.append(Phi)
    return(np.array(Data))

def GetAllPsiDipeptide(C1,NumResi=None,a=None,aC=None,aN=None,aCA=None,X=None):
    if a==None:
    	NumResi,a,aC,aN,aCA,X=GetPsiAtomIndicesAndPosition(C1)
    
    a0=aN[0]
    a1=aCA[0]#Corrected because resi 0 has no CA
    a2=aC[1]
    a3=aN[1]  
    x0=X[a0]
    x1=X[a1]
    x2=X[a2]
    x3=X[a3]
    Psi=Torsion(x0,x1,x2,x3)
    return(np.array([Psi]))

def GetAllPhiDipeptide(C1,NumResi=None,a=None,aC=None,aN=None,aCA=None,X=None):
    if a==None:
		NumResi,a,aC,aN,aCA,X=GetPhiAtomIndicesAndPosition(C1)
    a0=aC[0]
    a1=aN[0]
    a2=aCA[0]#Corrected because resi 0 has no CA
    a3=aC[1]
    x0=X[a0]
    x1=X[a1]
    x2=X[a2]
    x3=X[a3]
    Phi=Torsion(x0,x1,x2,x3)
    return(np.array([Phi]))

def Torsion(x0,x1,x2,x3,Degrees=True):
    """Calculate the signed dihedral angle between 4 positions."""
    #Calculate Bond Vectors b1, b2, b3
    b1=x1-x0
    b2=x2-x1
    b3=x3-x2

    #Calculate Normal Vectors c1,c2.  This numbering scheme is idiotic, so care.
    c1=np.cross(b2,b3)
    c2=np.cross(b1,b2)

    Arg1=np.dot(b1,c1)
    Arg1*=np.linalg.norm(b2)

    Arg2=np.dot(c2,c1)

    phi=np.arctan2(Arg1,Arg2)

    if Degrees==True:
        phi*=180./np.pi
    return(phi)

import numpy as np
import os
import KyleTools
from msmbuilder import Conformation,Trajectory,Project,Serializer

C1=Conformation.Conformation.load_from_pdb("./dipeptide.pdb")
P1=Project.Project.load_from_hdf("./ProjectInfo.h5")

NumResi,a,aC,aN,aCA,X=GetPhiAtomIndicesAndPosition(C1)
def GetPhi(X):
	return GetAllPhiDipeptide(C1,NumResi=NumResi,a=a,aC=aC,aN=aN,aCA=aCA,X=X)
Phi=P1.EvaluateObservableAcrossProject(GetPhi)
def GetPsi(X):
	return GetAllPsiDipeptide(C1,NumResi=NumResi,a=a,aC=aC,aN=aN,aCA=aCA,X=X)
Psi=P1.EvaluateObservableAcrossProject(GetPsi)

print(Phi)

Serializer.SaveData("./Phi.h5",Phi)
Serializer.SaveData("./Psi.h5",Psi)
