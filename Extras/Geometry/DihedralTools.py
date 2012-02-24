import HierarchicalClustering
import scipy

import numpy as np

def CalculateDihedralAngle(x0,x1,x2,x3):
    """Calculate the signed dihedral angle between 4 positions.  Result is in degrees."""
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

    phi*=180./np.pi

    return(phi)

def ExtractDihedralsFromArray(XYZ):
    """XYZ is an NumResi by 4 array, Nf is number of frames, Na number of atoms."""
    NumResi,NumTerms,NumCoords=XYZ.shape
    
    Data=[]
    for i in xrange(NumResi-1):
        Psi=CalculateDihedralAngle(*XYZ[i])
        Data.append(Psi)
    return(Data)

def GetAtomIndicesForChi1(C1):
    NResi=C1.GetNumberOfResidues()
    AID=C1.GetEnumeratedAtomID()
    RID=C1.GetEnumeratedResidueID()
    AName=C1["AtomNames"]
    Indices=[]
    for i in range(NResi):
        a0=np.where((AName=="N")&(RID==i))[0][0]
        a1=np.where((AName=="CA")&(RID==i))[0][0]
        try:
            a2=np.where((AName=="CB")&(RID==i))[0][0]
        except:
            a2=None
        try:
            a3=np.where(((AName=="CG")|(AName=="CG1"))&(RID==i))[0][0]
        except:
            a3=None
        if a3==None or a2==None:
            continue
        Indices.append([a0,a1,a2,a3])

    return(np.array(Indices))

def GetAtomIndicesForPhi(C1):
    NResi=C1.GetNumberOfResidues()
    AID=C1.GetEnumeratedAtomID()
    RID=C1.GetEnumeratedResidueID()
    AName=C1["AtomNames"]
    Indices=[]
    for i in range(NResi):
        try:
            a0=np.where((AName=="C")&(RID==i))[0][0]
            a1=np.where((AName=="N")&(RID==(i+1)))[0][0]
            a2=np.where((AName=="CA")&(RID==(i+1)))[0][0]
            a3=np.where((AName=="C")&(RID==(i+1)))[0][0]
        except:
            pass
        Indices.append([a0,a1,a2,a3])
    return(np.array(Indices))

def GetAtomIndicesForPsi(C1):
    NResi=C1.GetNumberOfResidues()
    AID=C1.GetEnumeratedAtomID()
    RID=C1.GetEnumeratedResidueID()
    AName=C1["AtomNames"]
    Indices=[]
    for i in range(NResi):
        try:
            a0=np.where((AName=="N")&(RID==i))[0][0]
            a1=np.where((AName=="CA")&(RID==i))[0][0]
            a2=np.where((AName=="C")&(RID==i))[0][0]
            a3=np.where((AName=="N")&(RID==(i+1)))[0][0]
        except:
            pass
        Indices.append([a0,a1,a2,a3])

    return(np.array(Indices))        

def GetAtomIndicesForTorsion(C1,Torsion):
    if Torsion=="Chi":
        return GetAtomIndicesForChi1(C1)
    if Torsion=="Phi":
        return GetAtomIndicesForPhi(C1)
    if Torsion=="Psi":
        return GetAtomIndicesForPsi(C1)
            
def GetTorsionFromConformation(C1,Torsion,Indices=None):
    Data=[]
    if Indices==None:
        Indices=GetAtomIndicesForTorsion(C1,Torsion)
    XYZ=C1["XYZ"][Indices]
    Data.append(ExtractDihedralsFromArray(XYZ))
    return(np.array(Data))

def GetTorsions(R1,C1,Torsion):
    Torsions=[]
    for i in range(R1["XYZList"].shape[0]):
	    C1["XYZ"]=R1["XYZList"][i]
	    if i==0:
                Indices=GetAtomIndicesForTorsion(C1,Torsion)
            Torsions.append(GetTorsionFromConformation(C1,Torsion,Indices=Indices))
    return((np.array((Torsions))).transpose())

def CalculatePairwiseTorsionMatrix(Project1,Dihedrals):
	"""This is the key (and rate limiting) calculation for SL clustering.  Calculate all pairwise distances in a dataset."""
	ConfListing,EC=HierarchicalClustering.ConstructConfListing(Project1)
	NumConf=ConfListing.max()+1

        n0=Dihedrals.shape[0]
        n1=Dihedrals.shape[1]
        DH=np.zeros((NumConf,2*Dihedrals.shape[-1]))
        for i in range(NumConf):
            x=np.repeat(Dihedrals[EC[i,0],EC[i,1]].flatten(),2)
            DH[i]=x
        DH[:,0::2]=np.cos(2*np.pi*DH[:,0::2]/360.)
        DH[:,1::2]=np.sin(2*np.pi*DH[:,1::2]/360.)
        DistanceMatrix=scipy.spatial.distance.cdist(DH,DH)                   
        return(DistanceMatrix)
