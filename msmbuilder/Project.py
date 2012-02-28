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

from msmbuilder import Conformation, Serializer, Clustering, Trajectory
from msmbuilder.DistanceMetric import RMSD

def GetUniqueRandomIntegers(MaxN,NumInt):
    """Get random numbers, with replacement."""
    x=np.random.random_integers(0,MaxN,NumInt)
    while len(np.unique(x))<NumInt:
        x2=np.random.random_integers(0,MaxN,NumInt-len(np.unique(x)))
        x=np.concatenate((np.unique(x),x2))
    return(x)

class Project(Serializer.Serializer):
    """The Project class controls access to a collection of trajectories."""
    
    def __init__(self,S=dict()):#TrajLengths,TrajFilePath,TrajFileBaseName,TrajFileType,ConfFilename):
        Serializer.Serializer.__init__(self,S)
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
    def GetNumTrajectories(self):
        """Return the number of trajectories in this project."""
        return(self["TrajLengths"].shape[0])
    def GetTrajFilename(self,TrajNumber):
        """Returns the filename of the Nth trajectory."""
        return(GetTrajFilename(self["TrajFilePath"],self["TrajFileBaseName"],self["TrajFileType"],TrajNumber))

    def LoadTraj(self,i):
        """Return a trajectory object of the ith trajectory."""
        return Trajectory.Trajectory.LoadTrajectoryFile(self.GetTrajFilename(i),Conf=self.Conf)

    def ReadFrame(self,WhichTraj,WhichFrame):
        """Read a single frame of a single trajectory."""
        return(Trajectory.Trajectory.ReadFrame(self.GetTrajFilename(WhichTraj),WhichFrame,Conf=self.Conf))
        
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

    def AssignProject(self, Generators, AtomIndices=None, WhichTrajs=None, AssFilename=None, AssRMSDFilename=None, CheckpointFile=None):
        """Given a set of Generators (a trajectory), assign each conformation in the dataset to Generators.
           
           If given the additional kwargs AssFilename, AssRMSDFilename, will attempt to load in those files
           and proceed from any usable data contained therein.
 
           If the kwarg 'CheckpointFile', is specified, then after each assigned 
           trajectory, we backup to that file, as well as that file name +'.RMSD'
        """

        # Initialize the arrays in which to store data
        if AtomIndices==None:
            AtomIndices=np.arange(Generators["XYZList"].shape[1])
        if WhichTrajs==None:
            WhichTrajs=np.arange(self["NumTrajs"])
        AssArray  = -1 * np.ones((len(WhichTrajs),max(self["TrajLengths"])),dtype='int')
        RMSDArray = -1 * np.ones((len(WhichTrajs),max(self["TrajLengths"])),dtype='float32')

        # If partially complete assignment files are present, load them in
        if AssFilename != None:
            if AssRMSDFilename != None:
                print "Found checkpoint files. Loading: %s, %s" % (AssFilename, AssRMSDFilename)
                assign = Serializer.Serializer.LoadFromHDF(AssFilename)
                rmsd   = Serializer.Serializer.LoadFromHDF(AssRMSDFilename)
                for i in xrange( self['NumTrajs'] ):
                    if len(np.where(assign['Data'][i]!=-1)[0]) != self['TrajLengths'][i]:
                        StartTraj=i
                        break
                    else:    
                        AssArray[i]  = assign['Data'][i]
                        RMSDArray[i] = rmsd['Data'][i]
                print "Found assignments up to traj number: %d" % StartTraj
            else:
                print "Error: No Assignment RMSD h5 file found! Cannot load checkpoint."
                StartTraj = 0
        else: StartTraj = 0

        # Iterate through each trajectory and assign
        Gens=Generators["XYZList"][:,AtomIndices].copy()
        PGens = RMSD.PrepareData(Gens) # RTM Feb6,2012
        for i in range( StartTraj, len(WhichTrajs) ):
            print("Assigning Trajectory %d"%WhichTrajs[i])
            
            R1=self.LoadTraj(WhichTrajs[i])
            #Ass,AssRMSD=Clustering.KCenters.Assign(Gens,R1["XYZList"][:,AtomIndices])
            Ass,AssRMSD=Clustering.KCenters.Assign(Gens, R1['XYZList'][:,AtomIndices], PGens) #RTM Feb6,2012 Avoid re-preparing gens

            AssArray[i][0:len(Ass)]=Ass
            RMSDArray[i][0:len(AssRMSD)]=AssRMSD
 
            # If requested, output checkpoint files
            if CheckpointFile != None:
                try:
                    os.remove(CheckpointFile)
                    os.remove(CheckpointFile+'.RMSD')
                except:
                    pass # I see no need to warn here, but it may be a good idea --TJL
                Serializer.SaveData(CheckpointFile, AssArray)
                Serializer.SaveData(CheckpointFile+'.RMSD', RMSDArray)
        
        return(AssArray,RMSDArray,WhichTrajs)
    
    def CalcRMSDAcrossProject(self,C1,ind0,ind1):
        """Calculate RMSD(C1,X) for all conformations X in the project.  Returns a numpy array that is padded with negative ones for all gaps in the data."""
        R1=Trajectory.Trajectory(C1)
        N1=self["TrajLengths"].shape[0]
        N2=max(self["TrajLengths"])
        RMSDArray=-1*np.ones((N1,N2))
        for i in range(N1):
            print(i)
            R1["XYZList"]=self.LoadTraj(i)["XYZList"][:]
            rmsd=R1.CalcRMSD(C1,ind0,ind1)
            del R1["XYZList"]
            RMSDArray[i,0:len(rmsd)]=rmsd
        return(RMSDArray)
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
    
    def ClusterProject(self,NumGen,AtomIndices=None,GetRandomConformations=False,NumConfsToGet=None,Which=None,Stride=1,SkipKCenters=False,DiscardFirstN=0,DiscardLastN=0,GlobalKMedoidIterations=0,LocalKMedoidIterations=0,RMSDCutoff=-1.,NormExponent=2.,StartingIndices=None):
        """Cluster the project into geometric states using either k-centers or (hybrid) k-medoids.

        Inputs:
        NumGen: the number of states.

        Keyword Arguments:
        AtomIndices: a numpy array of integers (zero-indexed!) specifying which atoms to use.
        GetRandomConformations: Set this to True to cluster using randomly selected conformations.
        NumConfsToGet: how many randomly selected conformations to use.  (Note: this a  guideline, and may contain duplicates.)
        Which: A numpy array of which trajectories to use.
        Stride: subsample your dataset with a given stride (e.g. x[::Stride])
        SkipKCenters: set this to True to skip the initial k-centers, instead using evenly sampled (in time) clusters.
        DiscardFirstN: throw away the first N frames of each trajectory.
        DiscardLastN: throw away the last N frames of each trajectory.
        GlobalKMedoidIterations: how many sweeps of 'global' k-medoids.
        LocalKMedoidIterations: how many sweeps of 'local' k-medoids.
        RMSDCutoff: terminate k-centers when state diameters reach this value.
        NormExponent: specify the exponent of the p-norm used in the k-medoid objective function.
        StartingIndices: specify the initial indices of your cluster centers, which will then be improved using k-medoids.

        Notes:
        The hybrid k-medoids algorithm here REJECTS all moves that increase the worse-case clustering error.
        
        For many systems, the protocol of k-centers then ~10 local k-medoids appears to give reasonable results.  
        """
        
        if GetRandomConformations==False:
            print("Getting Conformations at Stride=%d"%Stride)
            XYZ0=self.GetAllConformations(Stride=Stride,Which=Which,DiscardFirstN=DiscardFirstN,DiscardLastN=DiscardLastN)
        else:
            print("Getting %d Random Conformations"%NumConfsToGet)
            XYZ0=self.GetRandomConformations(NumConfsToGet,Which=Which)

        #Note that we initially load all the data (not just the cluster atoms).  This will be a problem if our data includes solvent.        
        if AtomIndices==None:
            AtomIndices=np.arange(len(XYZ0[0]))

        XYZ=XYZ0[:,AtomIndices]
        
        if not SkipKCenters:
            KCentersInd=Clustering.KCenters.Cluster(XYZ,NumGen,RMSDCutoff=RMSDCutoff)
        elif StartingIndices==None:
            #KCentersInd=GetUniqueRandomIntegers(len(XYZ)-1,NumGen)
            KCentersInd=np.linspace(0,len(XYZ)-1,NumGen).astype('int')
        else:
            KCentersInd=StartingIndices

        Ind=Clustering.HybridKMedoids.Cluster(XYZ,KCentersInd,NumIter=GlobalKMedoidIterations,NormExponent=NormExponent,LocalSearch=False)

        Ind=Clustering.HybridKMedoids.Cluster(XYZ,Ind,NumIter=LocalKMedoidIterations,NormExponent=NormExponent,LocalSearch=True)

        Trj=self.GetEmptyTrajectory()
        del XYZ#The next line might make a copy, so let's save some memory.
        Trj["XYZList"]=XYZ0[Ind].copy()
        del XYZ0#Just to be sure, let's get rid of this array.
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
    
    def SavePDBs(self,Ass,OutDir,NumConf,States=None):
        """Get random conformations from each state, then save them in directory OutDir.  Returns a Trajectory containing the conformations."""
        NumStates=max(Ass.flatten())+1
        try:
            os.mkdir(OutDir)
        except:
            pass
        R1=self.GetEmptyTrajectory()
        R2=self.GetEmptyTrajectory()

        if States==None:
            States=xrange(NumStates)
    
        for i in States:
            print(i)
            Outfile=OutDir+"/State%d-%d.pdb"%(i, NumConf-1)
            if os.path.exists(Outfile):
                print "  already done, skipping"
                continue
            R1["XYZList"]=self.GetRandomConfsFromState(Ass,i,NumConf)
            for j in xrange(NumConf):
                Outfile=OutDir+"/State%d-%d.pdb"%(i,j)
                print("Saving State %d Conf %d as %s"%(i,j,Outfile))
                R2["XYZList"]=np.array([R1["XYZList"][j]])
                R2.SaveToPDB(Outfile)
        return(R1)


    
def CountLocalTrajectories(TrajFilePath,TrajFileBaseName,TrajFileType):
    """In the current Project directory, look in the Trajectories Subdirectory and find the maximum N such that trj0.h5, trj1.h5, ..., trj[N-1].h5 all exist."""

    Unfinished=True
    i=0 

    while Unfinished:
        filename=GetTrajFilename(TrajFilePath,TrajFileBaseName,TrajFileType,i)
        if os.path.exists(filename):
            i += 1
        else:
            Unfinished=False
    return i

def CreateProjectFromDir(Filename="ProjectInfo.h5",TrajFilePath="./Trajectories/", TrajFileBaseName="trj",TrajFileType=".h5",ConfFilename=None,RunList=None,CloneList=None,NumGensList=None):
    """By default, the files should be of the form ./Trajectories/trj0.h5 ... ./Trajectories/trj[n].h5.
       Use optional arguments to change path, names, and filetypes."""

    if ConfFilename!=None:
        Conf=Conformation.Conformation.LoadFromPDB(ConfFilename)

    NumTraj=CountLocalTrajectories(TrajFilePath,TrajFileBaseName,TrajFileType)
    if NumTraj==0:
        print "No data found!  ERROR"
        return
    else: print "Found %d trajectories" % NumTraj

    LenList=[]
    for i in range(NumTraj):
        f=GetTrajFilename(TrajFilePath,TrajFileBaseName,TrajFileType,i)
        LenList.append(Trajectory.Trajectory.LoadTrajectoryFile(f,JustInspect=True,Conf=Conf)[0])

    DictContainer={"TrajLengths":np.array(LenList),"TrajFilePath":TrajFilePath,"TrajFileBaseName":TrajFileBaseName,"TrajFileType":TrajFileType,"ConfFilename":ConfFilename}
    if RunList!=None:
        DictContainer["RunList"]=RunList
    if CloneList!=None:
        DictContainer["CloneList"]=CloneList
    if NumGensList!=None:
        DictContainer["NumGensList"]=NumGensList
    P1=Project(DictContainer)
    if Filename!=None:
        P1.SaveToHDF(Filename)
    try:
        os.mkdir("./Data")
    except OSError:
        pass
    return P1

def CreateProjectInSitu( ConfFilename, TrajFilePath, TrajFileBaseName="trj", TrajFileType=".h5"):
    """ Generates a project object directly from a directory of trajectories, without writing
        any information to disk. This keeps unnecessary metadata to a minimum.

        Returns: Project object """

    Conf = Conformation.Conformation.LoadFromPDB(ConfFilename)
    NumTraj = CountLocalTrajectories(TrajFilePath,TrajFileBaseName,TrajFileType)

    if NumTraj == 0:
        print "Error: No data found in directory: %s" % TrajFilePath
        sys.exit(1)

    LenList=[]
    for i in range(NumTraj):
        f=GetTrajFilename(TrajFilePath,TrajFileBaseName,TrajFileType,i)
        LenList.append(Trajectory.Trajectory.LoadTrajectoryFile(f,JustInspect=True,Conf=Conf)[0])

    DictContainer = { "TrajLengths"      : np.array(LenList),
                      "TrajFilePath"     : TrajFilePath,
                      "TrajFileBaseName" : TrajFileBaseName,
                      "TrajFileType"     : TrajFileType,
                      "ConfFilename"     : ConfFilename }
    P1=Project(DictContainer)

    return P1

def GetTrajFilename(TrajFilePath,TrajFileBaseName,TrajFileType,TrajNumber):
    """This is a helper function to construct a filename for a trajectory file."""
    x=TrajFilePath+"/"+TrajFileBaseName+str(TrajNumber)+TrajFileType
    return(x)

def MergeMultipleAssignments(AssList,RMSDList,WhichTrajList):
    """Take a set of Assignment, RMSDList, and WhichTrajList arrays return one big array for each of Assignments, RMSDList, and WhichTrajs.  Used to merge parallelized assignment runs on clusters."""
    AssArray=[]
    RMSDArray=[]
    WhichTrajsArray=[]
    for i in range(len(AssList)):
        print(i,WhichTrajList[i],AssList[i])
        AssArray.append(AssList[i])
        RMSDArray.append(RMSDList[i])
        WhichTrajsArray.append(np.array(WhichTrajList[i]).reshape((-1)))


    AssArray=np.vstack(AssArray)
    RMSDArray=np.vstack(RMSDArray)
    WhichTrajsArray=np.concatenate(WhichTrajsArray).flatten()
    
    Ind=np.argsort(WhichTrajsArray)

    AssArray=AssArray[Ind]
    RMSDArray=RMSDArray[Ind]
    WhichTrajsArray=WhichTrajsArray[Ind]
    
    return(AssArray,RMSDArray,WhichTrajsArray)

