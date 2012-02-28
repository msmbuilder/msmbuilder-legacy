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

"""Tools for Hierarchical clustering.  You should also install the fastcluster library.
"""

import numpy as np
import scipy.cluster

from msmbuilder import DistanceMetric,Serializer,MSMLib
RMSD=DistanceMetric.RMSD

def ConstructConfListing(P1):
	"""Enumerate the conformations in the dataset and return mappings (arrays) from the conformation number <-> to the trajectory number, frame number."""
	EC=P1.EnumerateConformations()

	ConfListing=np.zeros((len(P1["TrajLengths"]),P1["TrajLengths"].max()),'int')-1

	for i,xy in enumerate(EC):
		ConfListing[xy[0],xy[1]]=i
	return(ConfListing,EC)


def CalculateRMSDMatrix(Project1,AtomIndices,WhichTrajs1=None,WhichTrajs2=None):
	"""This is the key (and rate limiting) calculation for SL clustering.  Calculate all pairwise distances in a dataset."""
	if WhichTrajs1==None or WhichTrajs2==None:
		WhichTrajs1=np.arange(Project1["NumTrajs"])
		WhichTrajs2=np.arange(Project1["NumTrajs"])

	ConfListing,EC=ConstructConfListing(Project1)
	NumConf=ConfListing.max()+1
	DistanceMatrix=np.zeros((NumConf,NumConf),dtype='float32')
	
        for i1 in WhichTrajs1:
		R1=Project1.LoadTraj(i1)
		RData1=RMSD.PrepareData(R1["XYZList"][:,AtomIndices])
		n1=len(R1["XYZList"])		
	        for i2 in WhichTrajs2:
			if i1 >i2:  continue
			WhereCurrentTraj=np.where(EC[:,0]==i2)[0]
			print("Trajs %d, %d"%(i1,i2))
			R2=Project1.LoadTraj(i2)
			n2=len(R1["XYZList"])
			RData2=RMSD.PrepareData(R2["XYZList"][:,AtomIndices])
			for j1 in xrange(n1):
				ConfInd1=ConfListing[i1,j1]
				RealR=RMSD.GetFastMultiDistance(RData1,RData2,j1)
				DistanceMatrix[ConfInd1,WhereCurrentTraj]=RealR
				DistanceMatrix[WhereCurrentTraj,ConfInd1]=RealR
			x=R2.pop("XYZList")
			del x,R2,RData2
		x=R1.pop("XYZList")
		del x,R1,RData1
	return(DistanceMatrix)

	
def AssignUsingScipy(Project1,ZMatrix,NumStates):
	ans=scipy.cluster.hierarchy.fcluster(ZMatrix,NumStates,criterion="maxclust")
	ConfListing,EC=ConstructConfListing(Project1)
	Assignments=0*ConfListing-1
	for k,State in enumerate(ans):
		i,j=EC[k]
		Assignments[i,j]=State
	#Subtract 1 to ensure that states begin with zero
	Assignments-=1
	Assignments[Assignments==-2]=-1
	return(Assignments)
