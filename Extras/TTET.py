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

"""Tools for simulating triplet-triplet energy transfer (TTET).
"""

import scipy.sparse
import numpy as np


def SimpleResDist(C1,X,i,j):
	"""Get the distance between residues i and j.  For residues with beta carbons, use the beta carbon distance.  Otherwise use alpha carbons."""
	r1=C1["ResidueNames"][C1["IndexList"][i][0]]
	r2=C1["ResidueNames"][C1["IndexList"][j][0]]
	a1="CB"
	a2="CB"
	if r1=="GLY": a1="CA"
	if r2=="GLY": a2="CA"
	i0=np.where((C1["AtomNames"]==a1)&(C1["ResidueID"]==i))
	i1=np.where((C1["AtomNames"]==a2)&(C1["ResidueID"]==j))
	return(np.linalg.norm(X[i0]-X[i1]))

def PeptideResDist(C1,X,i,j):
	"""Get the distance between residues i and j.  For residues with beta carbons, use the beta carbon distance.  Otherwise use alpha carbons."""
	r1=C1["ResidueNames"][C1["IndexList"][i][0]]
	r2=C1["ResidueNames"][C1["IndexList"][j][0]]
	a11="CK1"
	a12="CK8"
	a21="CG"
	a22="CK4"
	i11=np.where((C1["AtomNames"]==a11)&(C1["ResidueID"]==i))
	i12=np.where((C1["AtomNames"]==a12)&(C1["ResidueID"]==i))
	i21=np.where((C1["AtomNames"]==a21)&(C1["ResidueID"]==j))
	i22=np.where((C1["AtomNames"]==a22)&(C1["ResidueID"]==j))
	X1=.5*(X[i11]+X[i12])
	X2=.5*(X[i21]+X[i22])
	return(np.linalg.norm(X1-X2))

def KiefResDist(C1,X,i,j):
	"""Get the distance between resiudes i and j, picking atoms that correspond to the fluorophore attachment points in Kiefhaber PNAS 2010."""
	r1=C1["ResidueNames"][C1["IndexList"][i][0]]
	r2=C1["ResidueNames"][C1["IndexList"][j][0]]
	a1="CB"
	a2="CB"
	if r1=="GLY": a1="CA"
	if r2=="GLY": a2="CA"
	if i==0:
		a1="N"
	elif i==6:
		a1="NZ"
	elif i==34:
		a1="CB"
	if j==0:
		a2="N"
	elif j==6:
		a2="NZ"
	elif j==34:
		a2="CB"
	if (i==0 and j==34):
		a2="CB"
	if (i==34 and j==0):
		a1="CB"
	i0=np.where((C1["AtomNames"]==a1)&(C1["ResidueID"]==i))
	i1=np.where((C1["AtomNames"]==a2)&(C1["ResidueID"]==j))
	return(np.linalg.norm(X[i0]-X[i1]))


def GetSplitTransitionMatrix(T0,S):
	"""Take a transition matrix T0 and a set of coupling coefficients and construct a transition matrix for the joint TTET-conformational dynamics."""
	n=T0.shape[0]
	T1=scipy.sparse.lil_matrix((n*2,n*2))
	xind,yind=T0.nonzero()
	LightInd=np.where(S>0)[0]
	for k in range(len(xind)):
		i,j=xind[k],yind[k]
		T1[i,j]=T0[i,j]
		T1[i+n,j+n]=T0[i,j]
		T1[i,j+n]=0.0
		if i in LightInd:
			T1[i,j]=T0[i,j]*(1.0-S[i])
		T1[i+n,j]=0.0
	for i in LightInd:
		T1[i,n+i]=1.0*S[i]
	return(T1)


def MinDist(X,IndList1,IndList2):
	Alldlist=np.zeros((len(IndList1),len(IndList2)))
	for i0 in range(len(IndList1)):
		for i1 in range(len(IndList2)):
			x0=IndList1[i0]
			x1=IndList2[i1]
			Alldlist[i0,i1]=np.linalg.norm(X[x0]-X[x1])
	tdist=Alldlist.min(0).min(0)
	return(tdist)

def TTETVDWDist(C1,X,i,j):
	"""You need to set global variables TTET.Indi and TTET.Indj to the desired atoms.  This is a hack, but easier than designing an OO interface for just 1 lousy calculation."""
	ans=MinDist(X,Indi,Indj)
	return(ans)

def MinResDist(C1,X,i,j):
	"""Get the minimum distance between residues i and j."""
	Indi=C1["IndexList"][i]
	Indj=C1["IndexList"][j]
	ans=MinDist(X,Indi,Indj)
	return(ans)

def CalculateTransferCoefficientsCutoff(R1,i0,j0,TimeStep,TTETTimescale,Cutoff=1.0,tdist=None,ResDist=SimpleResDist):
	n=len(R1["XYZList"])
	if tdist==None:
		tdist=np.array([ResDist(R1,R1["XYZList"][i],i0,j0) for i in range(n)])
	S=np.zeros(len(R1["XYZList"]))
	S[np.where(tdist<Cutoff)]=1.0
	return(1.-np.exp(-S*TimeStep/TTETTimescale))

def CalculateTransferCoefficientsCutoffEntireProject(P1,Assignments,i0,j0,TimeStep,TTETTimescale,Cutoff=1.0,ResDist=SimpleResDist,UseMean=False):
	NumStates=max(Assignments.flatten())+1
	A=Assignments
	
	def DistanceFunctionWrapper(R1):
		n=R1["XYZList"].shape[0]
		distances=np.array([ResDist(R1,R1["XYZList"][i],i0,j0) for i in range(n)])
		return(distances)
	
	AllDistances=P1.EvaluateObservableAcrossProject(DistanceFunctionWrapper,ByTraj=True)
	Ind=np.where(AllDistances<Cutoff)
	AllDistances*=0.
	AllDistances[Ind]=1.

	if UseMean==True:
		SByState=np.array([1.-np.exp(-(TimeStep/TTETTimescale)*AllDistances[np.where(A==State)]).mean() for State in xrange(NumStates)])
	else:#Use the new, correct TTET model
		FractionActive=np.array([AllDistances[np.where(A==State)].mean() for State in xrange(NumStates)])
		f=FractionActive
		
		ko=FractionActive/TimeStep
		kc=1/float(TTETTimescale)
		e1=np.exp(-kc*TimeStep)
		e2=np.exp(-ko*TimeStep)
		A1=(ko-f*kc)/(kc-ko)
		A2=-1*(1-f)*kc / (kc-ko)

		SByState=1+A1*e1+A2*e2
	return(SByState)

def FastSimpleResDist(R1):
	"""Get the distance between residues i and j.  For residues with beta carbons, use the beta carbon distance.  Otherwise use alpha carbons."""
	print("In FastSimple")
	AtomIndices=[]
	NumFrames=len(R1["XYZList"])
	NumRes=len(R1["IndexList"])
	for i in range(NumRes):
		r1=R1["ResidueNames"][R1["IndexList"][i][0]]
		a1="CB"
		if r1=="GLY": a1="CA"
		i0=np.where((R1["AtomNames"]==a1)&(R1["ResidueID"]==i))[0][0]
		AtomIndices.append(i0)
	AtomIndices=np.array(AtomIndices)
	ResDistMatrix=np.zeros((NumFrames,NumRes,NumRes))
	for i in range(NumRes):
		for j in range(i,NumRes):				
			x1=R1["XYZList"][:,AtomIndices[i]]
			x2=R1["XYZList"][:,AtomIndices[j]]
			d=x1-x2				
			dist=((d**2).sum(1))**.5
			ResDistMatrix[:,i,j]=dist[:]
			ResDistMatrix[:,j,i]=dist[:]
	return(ResDistMatrix)

def CalculateAllTransferCoefficientsCutoffEntireProject(P1,Assignments,TimeStep,TTETTimescale,Cutoff=1.0):
	NumStates=max(Assignments.flatten())+1
	A=Assignments
	NumRes=len(P1.GetEmptyTrajectory()["IndexList"])
	print("In CalcAll")	
	AllDistances=P1.EvaluateObservableAcrossProject(FastSimpleResDist,ByTraj=True,ResultDim=(35,35))
	print("Done Getting Distances")
	for k in xrange(AllDistances.shape[0]):
		Ind=np.where(AllDistances[k]<Cutoff)
		AllDistances[k]*=0.
		AllDistances[k][Ind]=1.
	print("Done Calculating Cutoffs")
	SByState=np.zeros((NumStates,NumRes,NumRes),dtype='float32')

	print("Calculating Wherei")
	Wherei=[np.where(A==State) for State in xrange(NumStates)]
	print("Before Calculating TransferCoefficients")	
	for i in range(NumRes):
		for j in range(i,NumRes):
			print(i,j)
			FractionActive=np.array([AllDistances[:,:,i,j][Wherei[State]].mean() for State in xrange(NumStates)])
			f=FractionActive
		
			ko=FractionActive/TimeStep
			kc=1/float(TTETTimescale)
			e1=np.exp(-kc*TimeStep)
			e2=np.exp(-ko*TimeStep)
			A1=(ko-f*kc)/(kc-ko)
			A2=-1*(1-f)*kc / (kc-ko)

			S=1+A1*e1+A2*e2
			print(S.shape)
			print(S.min(),S.max(),S.mean(),S.std())
			SByState[:,i,j]=S
			SByState[:,j,i]=S
	return(SByState)

def MonitorTTET(T,NumSteps,EquilibriumPopulations,TransferCoefficients,StopAtValue=0.):
	n=T.shape[0]
	Tsplit=GetSplitTransitionMatrix(T,TransferCoefficients)
	Tlsplit=scipy.sparse.linalg.aslinearoperator(Tsplit.tocsr())
	X=np.zeros(2*n)
	X[0:n]=EquilibriumPopulations[0:n]
	TotalLightPopulation=[sum(X[0:n])]
	for i in range(NumSteps):
		X=Tlsplit.rmatvec(X)
		Value=sum(X[0:n])
		TotalLightPopulation.append(Value)
		print(i,Value)
		if Value < StopAtValue:
			break

	return(TotalLightPopulation,X)

