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

"""The Project controls interaction between MSMBuilder and a collection of MD trajectories.  
"""

import os
import numpy as np

from msmbuilder import Conformation
from msmbuilder.Serializer import Serializer
from msmbuilder.Trajectory import Trajectory
from msmbuilder import clustering

#def GetUniqueRandomIntegers(MaxN,NumInt):
#    """Get random numbers, with replacement."""
#    x=np.random.random_integers(0,MaxN,NumInt)
#    while len(np.unique(x))<NumInt:
#        x2=np.random.random_integers(0,MaxN,NumInt-len(np.unique(x)))
#        x=np.concatenate((np.unique(x),x2))
#    return(x)

class Project(Serializer):
    """The Project class controls access to a collection of trajectories."""
    
    def __init__(self,S=dict()):#TrajLengths,TrajFilePath,TrajFileBaseName,TrajFileType,ConfFilename):
        Serializer.__init__(self,S)
        if 'SerializerFilename' in S:
            self['ProjectRootDir'] = os.path.abspath(os.path.dirname(S['SerializerFilename']))
        else:
            self['ProjectRootDir'] = ''
            
        self["TrajLengths"]=np.array(S["TrajLengths"]).reshape((-1,))
        self["NumTrajs"]=len(self["TrajLengths"])
        self["TrajFileBaseName"]=S["TrajFileBaseName"]
        self["TrajFileType"]=S["TrajFileType"]
        
        if not S['ConfFilename'].startswith('/'):
            self['ConfFilename'] = os.path.join(self['ProjectRootDir'], S['ConfFilename'])
        else:
            self["ConfFilename"]=S["ConfFilename"]
        if not S['TrajFilePath'].startswith('/'):
            self['TrajFilePath'] = os.path.join(self['ProjectRootDir'], S['TrajFilePath'])
        else:
            self["TrajFilePath"]=S["TrajFilePath"]

        for key in ["RunList","CloneList","NumGensList"]:
            if key in S: self[key]=S[key]
        try:
            self.Conf=Conformation.Conformation.LoadFromPDB(self["ConfFilename"])
        except IOError:
            print("Could not find %s; trying current directory."%self["ConfFilename"])
            self.Conf=Conformation.Conformation.LoadFromPDB(os.path.basename(self["ConfFilename"]))
    
    @classmethod   
    def CountLocalTrajectories(cls, TrajFilePath, TrajFileBaseName, TrajFileType):
        """In the current Project directory, look in the Trajectories Subdirectory and find the maximum N such that trj0.h5, trj1.h5, ..., trj[N-1].h5 all exist."""

        Unfinished=True
        i=0 

        while Unfinished:
            filename = os.path.join(TrajFilePath, TrajFileBaseName + str(i) + TrajFileType)
            if os.path.exists(filename):
                i += 1
            else:
                Unfinished=False
        return i

    @classmethod
    def CreateProjectFromDir(cls, Filename="ProjectInfo.h5", TrajFilePath="./Trajectories/",
                             TrajFileBaseName="trj", TrajFileType=".h5", ConfFilename=None,
                             RunList=None, CloneList=None, NumGensList=None, initial_memory={}):
        """By default, the files should be of the form ./Trajectories/trj0.h5 ... ./Trajectories/trj[n].h5.
           Use optional arguments to change path, names, and filetypes."""

        if ConfFilename!=None:
            Conf=Conformation.Conformation.LoadFromPDB(ConfFilename)

        NumTraj = cls.CountLocalTrajectories(TrajFilePath,TrajFileBaseName,TrajFileType)
        if NumTraj==0:
            print "No data found!  ERROR"
            return
        else: print "Found %d trajectories" % NumTraj

        LenList=[]
        for i in range(NumTraj):
            f = os.path.join(TrajFilePath, TrajFileBaseName + str(i) + TrajFileType)
            LenList.append(Trajectory.LoadTrajectoryFile(f,JustInspect=True,Conf=Conf)[0])

        DictContainer={ "TrajLengths"      : np.array(LenList),
                        "TrajFilePath"     : TrajFilePath,
                        "TrajFileBaseName" : TrajFileBaseName,
                        "TrajFileType"     : TrajFileType,
                        "ConfFilename"     : ConfFilename,
                        "Memory"           : initial_memory }
        if RunList!=None:
            DictContainer["RunList"]=RunList
        if CloneList!=None:
            DictContainer["CloneList"]=CloneList
        if NumGensList!=None:
            DictContainer["NumGensList"]=NumGensList
        Project = cls(DictContainer)
        if Filename!=None:
            Project.SaveToHDF(Filename)
        
        try:
            os.mkdir("./Data")
        except OSError:
            pass
            
        return Project
    
    def GetNumTrajectories(self):
        """Return the number of trajectories in this project."""
        return(self["TrajLengths"].shape[0])
    
    def GetTrajFilename(self, TrajNumber):
        """Returns the filename of the Nth trajectory."""
        return os.path.join(self['TrajFilePath'], self['TrajFileBaseName'] + str(TrajNumber) +
                            self['TrajFileType'])

        
    def LoadTraj(self, i, stride=1):
        """Return a trajectory object of the ith trajectory."""
        return Trajectory.LoadTrajectoryFile(self.GetTrajFilename(i),Conf=self.Conf)[::stride]
    
    def EnumTrajs(self):
        """Convenience method: return an iterator over the trajectories (a generator)
        
        Example Usage:
        
        for traj in project.EnumTrajs():
            do_something(traj)
        
        
        """
        for i in xrange(self['NumTrajs']):
            yield Trajectory.LoadTrajectoryFile(self.GetTrajFilename(i), Conf=self.Conf)
        
    
    def ReadFrame(self,WhichTraj,WhichFrame):
        """Read a single frame of a single trajectory."""
        return(Trajectory.ReadFrame(self.GetTrajFilename(WhichTraj),WhichFrame,Conf=self.Conf))
    
    def EvaluateObservableAcrossProject(self,f,Stride=1,ByTraj=False,ResultDim=None):
        """Evaluate an observable function f for every conformation in a dataset.  Assumes that f returns a float.  Also initializes a matrix as negative ones and fills in f(X[i,j]) where X[i,j] is the jth conformation of the ith trajectory.  If ByTraj==True, evalulate f(R1) for each trajectory R1 in the dataset.  This allows you to dramatically speed up certain observable calculations."""
        k=0
        Which=np.arange(self["NumTrajs"])
        n1=len(Which)
        n2=max(self["TrajLengths"])
        if ResultDim!=None:
            print(n1,n2,ResultDim[0],ResultDim[1])
            Ans=np.ones((n1,n2,ResultDim[0],ResultDim[1]),dtype='float32')#Hack assuming rank 2--need to generalize this.
            for i in xrange(Ans.shape[0]):
                Ans[i,:]=-1
            print("constructed answer matrix")
        else:
            Ans=np.ones((n1,n2))
            Ans*=-1
        l=0
        for k in Which:
            print(k)
            R1=self.LoadTraj(k)
            R1["XYZList"][::Stride]
            if ByTraj==False:#Calculate a function of each array of XYZ coordinates
                for j,Z in enumerate(R1["XYZList"]):
                    value=f(Z)
                    Ans[l,j]=value
            else:#Calculate a function of the trajectory R1
                value=f(R1)
                Ans[l,:len(value)]=value
            l=l+1
        return(Ans)
    
    def GetEmptyTrajectory(self):
        """This creates a trajectory with the correct atoms and residues, but leaves the coordinate data empty (XYZList)."""
        Traj=self.LoadTraj(0)
        Traj.pop("XYZList")
        return(Traj)
    
    def EnumerateTrajs(self):
        """List which trajectories we have."""
        return(np.arange(self["NumTrajs"][0]))
    
    def GetRandomConfsFromEachState(self,Ass,NumStates,NumConf,Subsampling=1,JustGetIndices=False):
        """Given a set of Assignments data (Ass) with (NumStates) states, grab (NumConf) conformations from each state.  This returns a trajectory with all the conformations.  If you just want to know what trajectory / snapshot the conformations came from, set JustGetIndices=True."""
        Trj=self.GetEmptyTrajectory()
        Trj["XYZList"]=[]
        for i in range(NumStates):
            print("Getting Conformations For state %d"%i)
            XYZList=self.GetRandomConfsFromState(Ass,i,NumConf,Subsampling=Subsampling,JustGetIndices=JustGetIndices)
            Trj["XYZList"].append(XYZList)
        Trj["XYZList"]=np.array(Trj["XYZList"])
        if JustGetIndices==False:
            m0,m1,m2,m3=Trj["XYZList"].shape
            Trj["XYZList"]=Trj["XYZList"].reshape((m0*m1,m2,m3))
        else:
            Trj=Trj["XYZList"]
        return(Trj)
    
    def GetRandomConfsFromState(self,Ass,state,NumConf,Subsampling=1,JustGetIndices=False):
        """Given a set of assignments (Ass: a numpy array), grab random conformations from this state.  If you would rather just grab indices of conformations, set JustGetIndices=True.  Note that This function assumes that we have the assignments loaded in memory, which we may want to fix later on."""
        ind=np.where(Ass==state)
        # print(ind) # TJL turned off excessive verbosity
        M=len(ind[0])
        XYZList=[]
        try:
            RND=np.random.random_integers(0,M-1,NumConf)
            for r in RND:
                k1=ind[0][r]
                k2=ind[1][r]*Subsampling
                # print("k1,k2",k1,k2)
                if not JustGetIndices:
                    XYZList.append(self.ReadFrame(k1,k2))
                else:
                    XYZList.append((k1,k2))
        except ValueError:
            pass
        return(np.array(XYZList))
    
    def GetConformations(self,Which):
        """Get a trajectory containing the conformations specified by Which. Which should be a 2d array, such that Which[i]=x,y. x is the trajectory number, and y is the frame number."""
        N1=len(Which)
        Trj=self.GetEmptyTrajectory()
        
        NumAtoms=len(Trj["AtomNames"])
        NumAxes=3
        Current=0
        
        Trj["XYZList"]=np.zeros((0,NumAtoms,NumAxes),dtype='float32')
        for i in range(N1):
            X=self.ReadFrame(Which[i,0],Which[i,1])
            Trj["XYZList"].resize((Current+1,NumAtoms,NumAxes))
            Trj["XYZList"][Current]=X
            Current+=1
            
        Trj["XYZList"]=np.array(Trj["XYZList"],dtype='float32')
        return(Trj)
    
    def GetAllConformations(self,Stride=1,Which=None,AtomIndices=None,DiscardFirstN=0,DiscardLastN=0):
        """Get all conformations from this dataset.  Setting Stride=N allows you to select every Nth conformation.  Setting Which allows you to specify which trajectories to grab from.  Setting AtomIndices will select coordinates from specific atoms."""
        if AtomIndices==None:
            Traj=self.LoadTraj(0)
            AtomIndices=np.arange(Traj["XYZList"].shape[1])
        
        XYZList=[]
        if Which==None:
            Which=np.arange(self["NumTrajs"])
        
        NumAtoms=len(AtomIndices)
        NumAxes=3
        
        XYZList=np.zeros((0,NumAtoms,NumAxes),dtype='float32')
        Current=0
        for i in Which:
            Traj=self.LoadTraj(i)
            TrajLen=Traj["XYZList"].shape[0]
            
            X=Traj["XYZList"][DiscardFirstN:(TrajLen-DiscardLastN):Stride][:,AtomIndices]
            
            XYZList.resize((Current+len(X),NumAtoms,NumAxes))
            XYZList[Current:Current+len(X)]=X
            Current+=len(X)
            del Traj["XYZList"]
            del X
                           
        return(np.array(XYZList,dtype='float32'))
    
    def EnumerateConformations(self,Which=None):
        """Return a 2d array that enumerates the indices of all conformations in the dataset.  Data[k] = x,y tells us that conformation k in this dataset belongs to trajectory x and is the yth conformation in that trajectory."""
        iN=[]
        if Which==None:
            Which=np.arange(self["NumTrajs"])
        for i in Which:
            N=self["TrajLengths"][i]
            iN.append((i,N))
        SumN=sum([x[1] for x in iN])
        Data=np.zeros((SumN,2),dtype='int')
        k=0
        for x in iN:
            i=x[0]
            N=x[1]
            Data[k:k+N,0]=i
            Data[k:k+N,1]=range(N)
            k=k+N
        return(Data)
    
    def GetRandomConformations(self,NumConfs,Which=None):
        """Now without replacement!"""
        ConfIndices=self.EnumerateConformations(Which=Which)
        R1=np.random.random_integers(0,len(ConfIndices)-1,NumConfs)
        R2=[]
        for r in R1:
            if r not in R2:
                R2.append(r)
        RandIndices=ConfIndices[R2]
        Trj=self.GetConformations(RandIndices)
        return(Trj["XYZList"])


#def MergeMultipleAssignments(AssList,RMSDList,WhichTrajList):
#    """Take a set of Assignment, RMSDList, and WhichTrajList arrays return one big array for each of Assignments, RMSDList, and WhichTrajs.  Used to merge parallelized assignment runs on clusters."""
#    AssArray=[]
#    RMSDArray=[]
#    WhichTrajsArray=[]
#    for i in range(len(AssList)):
#        print(i,WhichTrajList[i],AssList[i])
#        AssArray.append(AssList[i])
##        RMSDArray.append(RMSDList[i])
#        WhichTrajsArray.append(np.array(WhichTrajList[i]).reshape((-1)))#
##
#
#    AssArray=np.vstack(AssArray)
#    RMSDArray=np.vstack(RMSDArray)
#    WhichTrajsArray=np.concatenate(WhichTrajsArray).flatten()
#    
#    Ind=np.argsort(WhichTrajsArray)
#
#    AssArray=AssArray[Ind]
#    RMSDArray=RMSDArray[Ind]
#    WhichTrajsArray=WhichTrajsArray[Ind]
#    
#    return(AssArray,RMSDArray,WhichTrajsArray)

