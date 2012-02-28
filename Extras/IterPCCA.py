
import numpy as np
import scipy.io, scipy.sparse
from msmbuilder import Serializer,MSMLib
import HierarchicalClustering
import os

def BuildMicro(MicroAss):
    MicroCounts=MSMLib.GetCountMatrixFromAssignments(MicroAss)
    MicroCounts,Map=MSMLib.ErgodicTrim(MicroCounts)
    MSMLib.ApplyMappingToAssignments(MicroAss,Map)
    
    MicroCounts = MSMLib.IterativeDetailedBalance(MicroCounts,Prior=0.0)
    T = MSMLib.EstimateTransitionMatrix(MicroCounts)
    l,v=MSMLib.GetEigenvectors(T,10)
    print(-50/np.log(l))

    return T,MicroCounts

class PCCAIssue(Exception):
	pass

def BuildMacro(MicroAss,T,NumMacro):
#    if os.path.exists("./ass.h5"):
#        os.remove("./ass.h5")
#    Serializer.SaveData("./ass.h5",MicroAss)
    Map = MSMLib.PCCA_Simplex(T, NumMacro, doMinimization=True)
    print("Unique map = ",len(np.unique(Map)))
    MacroAss=MicroAss.copy()
    MSMLib.ApplyMappingToAssignments(MacroAss,Map)
    if len(np.unique(Map))<NumMacro:
        print("Warning: PCCA+ detected only %d states when %d were desired."%(len(np.unique(Map)),NumMacro))
        MSMLib.RenumberStates(MacroAss)
        MacroCounts=MSMLib.GetCountMatrixFromAssignments(MacroAss,NumStates=MacroAss.max()+1)
        raise PCCAIssue()
    
    MacroCounts=MSMLib.GetCountMatrixFromAssignments(MacroAss,NumStates=MacroAss.max()+1)

    print("Observed # Macro states = ",MacroAss.max()+1)
    print(MacroCounts.toarray())
        
    #MacroCounts,Map=MSMLib.ErgodicTrim(MacroCounts)
    #MSMLib.ApplyMappingToAssignments(MacroAss,Map)

    #print("After Trim, bserved # Macro states = ",MacroAss.max()+1)
    #print(MacroCounts.toarray())

    MacroCounts = MSMLib.IterativeDetailedBalance(MacroCounts,Prior=0.0)

    return MacroAss,MacroCounts,Map

def Trim(MicroCounts,MacroCounts,MicroAss,MacroAss,Map,Epsilon,Merge=False):

    Pi=np.array(MacroCounts.sum(0)).flatten()
    Pi/=Pi.sum()
    print("Macrostate Pops")
    print(Pi)
    Ind=np.where(Pi<Epsilon)[0]
    print("Trimming discards %f of the population."%Pi[Ind].sum())
    
    MicroCounts=MicroCounts.toarray()
    MacroCounts=MacroCounts.toarray()

    if Merge==False:
        for i in Ind:
            MacroCounts[:,i]=0.
            MacroCounts[i,:]=0.
            for j in np.where(Map==i)[0]:
                MicroCounts[:,j]=0.
                MicroCounts[j,:]=0.
    else:
        NewMap=np.arange(MicroCounts.shape[0],dtype='int')
        for i in Ind:
            MacroCounts[:,i]=0.
            MacroCounts[i,:]=0.

            j=np.where(Map==i)[0]
            if len(j) >0:
                j=j[0]
            else:
                continue
            CTemp=MicroCounts[j,j]
            MicroCounts[j,j]=0.0
            MergeInd=np.argmax(MicroCounts[:,j])
            MicroCounts[j,j]=CTemp
            NewMap[j]=MergeInd
        #We merge "slow"  microstates with low counts to their nearest neighbor.
        MSMLib.ApplyMappingToAssignments(MicroAss,NewMap)
        
    MicroCounts=scipy.sparse.csr_matrix(MicroCounts)
    MicroCounts,Map=MSMLib.ErgodicTrim(MicroCounts)
    MSMLib.ApplyMappingToAssignments(MicroAss,Map)

    MacroCounts=scipy.sparse.csr_matrix(MacroCounts)
    MacroCounts,Map=MSMLib.ErgodicTrim(MacroCounts)
    MSMLib.ApplyMappingToAssignments(MacroAss,Map)
    print("Observed # Macro states = ",MacroAss.max()+1)


def IterativePCCAPlus(MicroAss,NumMacro,Epsilon,MaxIter=10):

    for k in range(MaxIter):
        print("*********************************")

        T,MicroCounts=BuildMicro(MicroAss)
        try:
            MacroAss,MacroCounts,Map=BuildMacro(MicroAss,T,NumMacro)
        except PCCAIssue:
            return MacroAss
        
        Trim(MicroCounts,MacroCounts,MicroAss,MacroAss,Map,Epsilon,MaxIter)

        CurrentNumMacro=MacroAss.max()+1
        if CurrentNumMacro==NumMacro:
            break
        
    return MacroAss

def TrimLowCounts(CMicro,CMacro,MicroAss,MacroAss,Map,Epsilon=.02):

    CMicro=CMicro.toarray()
    CMacro=CMacro.toarray()

    pi=CMacro.sum(0)
    pi/=pi.sum()

    Ind=np.where(pi<Epsilon)[0]
    print("Trimming discards %f of the population."%pi[Ind].sum())
    
    for i in Ind:
        CMacro[:,i]=0.
        CMacro[i,:]=0.

    for i in Ind:
	for j in np.where(Map==i)[0]:
            CMicro[:,j]=0.
            CMicro[j,:]=0.

    CMicro=scipy.sparse.csr_matrix(CMicro)
    CMacro=scipy.sparse.csr_matrix(CMacro)

    CMicro2,Unused=MSMLib.ErgodicTrim(CMicro,MicroAss)
    CMacro2,Unused=MSMLib.ErgodicTrim(CMacro,MacroAss)

def CoarsenLowCounts(CMicro,CMacro,MicroAss,MacroAss,Map,PairwiseRMSD,CL,EC,Epsilon=.02):
    """Coarsen the resolution of the microstate model at slow states with low population.
    """

    CMicro=CMicro.toarray()
    CMacro=CMacro.toarray()

    pi=CMacro.sum(0)
    pi/=pi.sum()

    Ind=np.where(pi<Epsilon)[0]
    
    MedoidIndices=FindMedoids(MicroAss,PairwiseRMSD,CL,EC)
    MedoidRMSD=PairwiseRMSD[MedoidIndices][:,MedoidIndices]
    MedoidRMSD+=1000*np.eye(MedoidRMSD.shape[0])

    i=Ind[0]
    j=np.where(Map==i)[0][0]
    jmedoid=MedoidIndices[j]
    NeighborState=np.argmin(MedoidRMSD[j])
    print(j,jmedoid,MedoidRMSD.shape,NeighborState)
    MicroAss[MicroAss==j]=NeighborState
    MSMLib.RenumberStates(MicroAss)

def FindMedoids(Ass,RMSD,CL,EC):
    NumStates=Ass.max()+1
    Medoids=np.zeros(NumStates,dtype='int')
    for i in range(NumStates):
        R=RMSD[CL[Ass==i]][:,CL[Ass==i]]
        Ind=np.argmin((R**2).sum(1))
        Medoids[i]=CL[Ass==i][Ind]
    return Medoids

def ReconstructMacroMap(Am,AM):
    NumMicro=Am.max()+1
    NumMacro=AM.max()+1
    Map=np.zeros(NumMicro,dtype='int')
    for i in range(NumMicro):
        Macro=AM[Am==i][0]
        Map[i]=Macro
    return Map

def CoarsenPCCAPlus(Am,NumMacro,PairwiseRMSD,CL,EC,MaxIter=20,Epsilon=0.02,Symmetrize=False):
    for i in range(MaxIter):
	AM=Am.copy()
	Cm=MSMLib.GetCountMatrixFromAssignments(Am)
	Cm=MSMLib.ErgodicTrim(Cm,Am)[0]
        if Symmetrize==True:
            Cm=0.5*(Cm+Cm.transpose())
	Cm=MSMLib.IterativeDetailedBalance(Cm)
	Tm=MSMLib.EstimateTransitionMatrix(Cm)
	Map=MSMLib.PCCA_Simplex(Tm, NumMacro, True)
	MSMLib.ApplyMappingToAssignments(AM,Map)
	CM=MSMLib.GetCountMatrixFromAssignments(AM)
	CM=MSMLib.ErgodicTrim(CM,AM)[0]
	CM=MSMLib.IterativeDetailedBalance(CM)
	Map=ReconstructMacroMap(Am,AM)
        p=np.array(CM.sum(0)).flatten()
        p/=p.sum()
        print("Current Number Microstates: ", Tm.shape[0])
        print("Number Macrostates detected by PCCA+", len(p))
        print(p)
        if p.min()>=Epsilon:
            break
        CoarsenLowCounts(Cm,CM,Am,AM,Map,PairwiseRMSD,CL,EC,Epsilon=Epsilon)

    AM=Am.copy()
    Cm=MSMLib.GetCountMatrixFromAssignments(Am)
    Cm=MSMLib.ErgodicTrim(Cm,Am)[0]
    if Symmetrize==True:
        Cm=0.5*(Cm+Cm.transpose())
    Cm=MSMLib.IterativeDetailedBalance(Cm)
    Tm=MSMLib.EstimateTransitionMatrix(Cm)
    Map=MSMLib.PCCA_Simplex(Tm, NumMacro, True)
    MSMLib.ApplyMappingToAssignments(AM,Map)
    CM=MSMLib.GetCountMatrixFromAssignments(AM)
    CM=MSMLib.ErgodicTrim(CM,AM)[0]
    CM=MSMLib.IterativeDetailedBalance(CM)

