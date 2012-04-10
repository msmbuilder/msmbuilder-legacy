import sys
import scipy.io
from Emsmbuilder import Serializer, MSMLib, SCRE
import numpy as np

DataDir="./Macro4/"
Tau = 1

Ass=Serializer.LoadData("./%s/Assignments.Fixed.h5"%DataDir)
C=scipy.io.mmread(DataDir+"tCounts.mtx").toarray()
Pi=loadtxt(DataDir+"Populations.dat")
C/=C.sum()
NumStates=Ass.max()+1

T=MSMLib.EstimateTransitionMatrix(C)
K0=SCRE.ConvertTIntoK(T)

M,X=SCRE.GetParamMapping(K0)

LagTimeList=arange(1,12)
KList=[]
CountsList=[]
for LagTime in LagTimeList:
	K=K0.copy() * float(LagTime)
	C0=MSMLib.GetCountMatrixFromAssignments(Ass,LagTime=LagTime).toarray()
	Counts=C0.sum(1)
	Counts/=LagTime
	X2=SCRE.MaximizeRateLikelihood(X,M,Pi,C0,K)
	K=SCRE.ConstructRateFromParams(X2,M,Pi,K)
	K/=(LagTime)
	KList.append(K)
	CountsList.append(Counts)

KList=array(KList)
SCRE.PlotRates(KList,LagTimeList,CountsList=CountsList,Tau=Tau)

SCRE.FixEntry(M,X,Pi,K0,3,1,KList[6][3,1])
SCRE.FixEntry(M,X,Pi,K0,1,0,KList[4][1,0])
SCRE.FixEntry(M,X,Pi,K0,3,0,KList[8][3,0])

LagTimeList=arange(1,25)
KList=[]
CountsList=[]
for LagTime in LagTimeList:
	K=K0.copy() * float(LagTime)
	C0=MSMLib.GetCountMatrixFromAssignments(Ass,LagTime=LagTime).toarray()
	Counts=C0.sum(1)
	Counts/=LagTime
	X2=SCRE.MaximizeRateLikelihood(X,M,Pi,C0,K)
	K=SCRE.ConstructRateFromParams(X2,M,Pi,K)
	K/=(LagTime)
	KList.append(K)
	CountsList.append(Counts)

KList=array(KList)
SCRE.PlotRates(KList,LagTimeList,CountsList=CountsList,Tau=Tau)

K=KList[13]

T=scipy.linalg.matfuncs.expm(K)
savetxt(DataDir+"/Rate.dat",K)
scipy.io.mmwrite(DataDir+"/tProb.mtx.tl",T)
