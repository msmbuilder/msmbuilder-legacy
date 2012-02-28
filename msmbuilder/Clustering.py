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

"""K-Centers and (hybrid) K-Medoids Clustering.

Notes:
This module contains code for clustering and assigning.
The easiest way to cluster is to use the command-line scripts in the Scripts directory.
Ordinary users will not directly use this code.  
"""
import numpy as np

from msmbuilder import DistanceMetric
RMSD=DistanceMetric.RMSD

def pNorm(Data,p=2):
    """Returns the p-norm of a Numpy array containing XYZ coordinates."""
    if p =="max":
        return Data.max()
    else:
        p=float(p)
        n=float(Data.shape[0])
        return ((Data**p).sum()/n)**(1/p)

class Clusterer:
    """Knows how to assign data to nearest cluster centers."""
    def Assign(self,Generators,XYZData,PreparedGens=None):
        """Assign data to Generators. Return arrays containing assignments and distances to cluster centers.

        Inputs:
        Generators -- XYZ coordinates of cluster centers.
        XYZData -- XYZ coordinates of data to assign.

        Keyword Arguments:
        PreparedGens -- Optionally include a DistanceMetric.RMSDContainer object.  This is used to avoid cost of repeated data initialization.
        """

        NumConfs=len(XYZData)

        Assignments=np.zeros(len(XYZData),'int')
        RMSDToGenerators=np.zeros(len(XYZData),'float32')

        if PreparedGens==None:
            PreparedGens=RMSD.PrepareData(Generators)
        else:
            #print("Using pre-calculated RMSD calculations for generators.")
            PreparedGens.CheckCentered()        

        PreparedData=RMSD.PrepareData(XYZData)

        for k in xrange(NumConfs):
            # print("Assigning conformation %d"%k) # --TJL commented for verbosity
            BestRMSD=np.inf
            BestInd=np.inf
            RMSDList=RMSD.GetFastMultiDistance(PreparedData,PreparedGens,k)
            BestInd=np.argmin(RMSDList)
            BestRMSD=RMSDList[BestInd]

            Assignments[k]=BestInd
            RMSDToGenerators[k]=BestRMSD
        return(Assignments,RMSDToGenerators)

class KCentersClusterer(Clusterer):
    """Can Assign and Cluster data using K-Centers algorithm."""
    def Cluster(self,XYZData,NumGen,Seed=0,RMSDCutoff=-1.):
        """Cluster data using k-centers and return the indices of the generators.

        Inputs:
        XYZData -- a numpy array containing conformations to be clustered.
        NumGen  -- the (maximum) number of clusters to return.

        Keyward arguments:
        Seed -- which frame of XYZData is used as starting cluster.  Default 0.
        RMSDCutoff -- terminate when all data lies less than RMSDCutoff from the nearest generator.  Default -1 [nm].
        """

        n0,n1,n2=XYZData.shape

        PreparedData=RMSD.PrepareData(XYZData)

        GeneratorIndices=[Seed]

        RMSDList=np.ones(n0)*np.inf
        for k in xrange(NumGen-1):
            print("Finding Generator %d"%(k+1))
            NewRMSDList=RMSD.GetFastMultiDistance(PreparedData,PreparedData,GeneratorIndices[k])
            RMSDList[np.where(NewRMSDList<RMSDList)]=NewRMSDList[np.where(NewRMSDList<RMSDList)]
            NewInd=np.argmax(RMSDList)
            if RMSDList[NewInd] < RMSDCutoff:
                break#The current generators are good enough; do not add another one.
            GeneratorIndices.append(NewInd)
        return(GeneratorIndices)

class HybridKMedoidsClusterer(Clusterer):
    """Can Cluster using hybrid K-Medoids.  Can also assign."""
    def EnsureNonemptyStates(self,PreparedGens,PreparedData,GeneratorIndices,n0,Assignments,NumGen,RMSDToCenters):
        """Assign data to generators and ensure all states have at least 1 conformation.  Empty states replaced by randomly select confs.

        Inputs:
        PreparedGens -- A DistanceMetric.TheoData object for the generators.
        PreparedData -- A DistanceMetric.TheoData object for the conformations.
        GeneratorIndices -- A list of frame indices corresponding to the generators.
        n0 -- The number of conformations in the dataset.
        Assignments -- An initial set of assignments (numpy array of integers).
        NumGen -- The number of clusters.
        RMSDToCenters -- The distance from each conformation to the closest cluster center.

        Notes:
        This function is primarily used during k-medoids clustering.
        """
        StillHaveEmptyGenerators=True
        while StillHaveEmptyGenerators==True:
            print("Assign data to generators and ensure that no generators are empty.")
            PreparedGens.SetData(PreparedData.GetData()[GeneratorIndices],PreparedData.GetG()[GeneratorIndices])
            for k in xrange(n0):
                RMSDList=RMSD.GetFastMultiDistance(PreparedData,PreparedGens,k)
                Assignments[k]=np.argmin(RMSDList)
                RMSDToCenters[k]=RMSDList[Assignments[k]]

            EmptyGenIndices=np.setdiff1d(np.arange(NumGen),np.unique(Assignments))
            print(EmptyGenIndices)
            print(GeneratorIndices[np.array(EmptyGenIndices)])
            for Em in EmptyGenIndices:
                TrialGen=np.random.random_integers(0,n0-1)
                GeneratorIndices[Em]=TrialGen
            if len(EmptyGenIndices)==0:
                    StillHaveEmptyGenerators=False

    def Cluster(self,XYZData,InitialGeneratorIndices,NumIter=10,NormExponent=2.,LocalSearch=False,TooCloseCutoff=.0001,IgnoreMaxObjective=False):
        """Cluster data using PAM-like (hybrid) k-medoids and return the indices of the generators.

        Inputs:
        XYZData -- a numpy array containing conformations to be clustered.
        InitialGeneratorIndices -- a list of frame indices pointing to the starting generators.  

        Keyward arguments:
        NumIter -- the number of iterations to perform.  Default 10.
        NormExponent -- the exponent of the p-norm in the objective function; changes weight of outliers.  Default 2.
        LocalSearch -- Restrict generator swaps to conformations assigned to a given state.  Default False.
        TooCloseCutoff -- reject moves when distance is less than this.  Default 0.0001 [nm].
        IgnoreMaxObjective -- Set this to True to accept moves that increase the worst-case clustering error.  When False, this function performs 'hybrid' k-medoids.  Default False.
        """

        n0,n1,n2=XYZData.shape

        if NumIter<=0:
            print("Skipping Medoid Step")
            return(InitialGeneratorIndices)
        print("Exponent of p-Norm  = %f"%NormExponent)

        PreparedData=RMSD.PrepareData(XYZData)
        PreparedGens=RMSD.PrepareData(XYZData[InitialGeneratorIndices])

        NumGen=len(InitialGeneratorIndices)
        GeneratorIndices=np.array(InitialGeneratorIndices).copy()

        Assignments=-1*np.ones(n0)
        RMSDToCenters=-1*np.ones(n0,dtype='float32')

        self.EnsureNonemptyStates(PreparedGens,PreparedData,GeneratorIndices,n0,Assignments,NumGen,RMSDToCenters)

        ObjectiveFunction=pNorm(RMSDToCenters,p=NormExponent)
        OldMaxNorm=pNorm(RMSDToCenters,p="max")

        FirstObjectiveFunction=ObjectiveFunction
        for k in xrange(NumIter):
            for WhichInd in xrange(NumGen):
                if LocalSearch==False:
                    TrialInd=np.random.random_integers(0,n0-1)
                else:
                    AssignedToState=np.where(Assignments==WhichInd)[0]
                    NumAssigned=len(AssignedToState)
                    TrialInd=np.random.random_integers(0,NumAssigned-1)
                    TrialInd=AssignedToState[TrialInd]

                print("Sweep %d: Try swapping Generator %d (Conf %d) with Conf %d"%(k,WhichInd,GeneratorIndices[WhichInd],TrialInd))
                if RMSDToCenters[TrialInd]<TooCloseCutoff:
                    print("Reject move: this conformation is too close to an existing generator.")
                    continue
                NewGeneratorIndices=GeneratorIndices.copy()
                NewGeneratorIndices[WhichInd]=TrialInd

                PreparedGens.SetData(PreparedData.GetData()[NewGeneratorIndices],PreparedData.GetG()[NewGeneratorIndices])

                NewAssignments=Assignments.copy()
                NewRMSDToCenters=RMSDToCenters.copy()

                RMSDToTrialGen=RMSD.GetFastMultiDistance(PreparedData,PreparedData,TrialInd)

                AssignedToNewState=np.where(RMSDToTrialGen<RMSDToCenters)[0]
                NewAssignments[AssignedToNewState]=WhichInd
                NewRMSDToCenters[AssignedToNewState]=RMSDToTrialGen[AssignedToNewState]

                AmbiguousAssigned=np.where((Assignments==WhichInd)&(RMSDToTrialGen>RMSDToCenters))[0]
                for l in AmbiguousAssigned:
                    RMSDList2=RMSD.GetFastMultiDistance(PreparedData,PreparedGens,l)
                    NewAssignments[l]=np.argmin(RMSDList2)
                    NewRMSDToCenters[l]=RMSDList2[NewAssignments[l]]

                NewObjectiveFunction=pNorm(NewRMSDToCenters,p=NormExponent)
                NewMaxNorm=pNorm(NewRMSDToCenters,p="max")
                print("New f = %f, Old f = %f, Old Max Norm = %f,(Values report root mean square distance from assigned center, in nm)"%(NewObjectiveFunction, ObjectiveFunction,OldMaxNorm))
                #if (NewObjectiveFunction < ObjectiveFunction) or (NewObjectiveFunction==ObjectiveFunction and New2Norm<Old2Norm):
                if NewObjectiveFunction < ObjectiveFunction:
                    if NewMaxNorm<=OldMaxNorm or IgnoreMaxObjective==True:
                        print("Accept")
                        GeneratorIndices=NewGeneratorIndices
                        Assignments=NewAssignments
                        RMSDToCenters=NewRMSDToCenters
                        ObjectiveFunction=NewObjectiveFunction
                        OldMaxNorm=NewMaxNorm


        print("Starting and Final Objective Functions: %f %f"%(FirstObjectiveFunction,ObjectiveFunction))
        return(GeneratorIndices)

KCenters=KCentersClusterer()
HybridKMedoids=HybridKMedoidsClusterer()
