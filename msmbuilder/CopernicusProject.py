import Conformation
import os
import numpy as np

from msmbuilder import Conformation, Project, Trajectory
from msmbuilder.ssaCalculator import ssaCalculator

class CopernicusProject(Project.Project):
    def GetTrajFilename(self,TrajNumber):
        """Returns the filename of the Nth trajectory."""

        if TrajNumber >= self["NumTrajs"]:
            print "ERROR: no traj number %d" % TrajNumber
            return

        i = 0
        while i+1 < self["NumTrajs"] and TrajNumber >= self["PartsLenList"][i+1]:
            i += 1
        PartNum = TrajNumber - self["PartsLenList"][i]

        return(self["FileList"][i][PartNum])

    def AssignProject(self,Generators,AtomIndices=None,WhichTrajs=None):
        ass, rmsd, w = Project.Project.AssignProject(self, Generators, AtomIndices=AtomIndices, WhichTrajs=WhichTrajs)
        return ass,rmsd,w

    def ClusterProject(self,AtomIndices=None,XTCOut=None,NumGen=None, Stride=None):

        TotNumConfs = self["TrajLengths"].sum()
        if Stride == None:
            Stride = 10

        if NumGen == None:
            NumGen = TotNumConfs / Stride / 10

        Gens = Project.Project.ClusterProject(self,NumGen,AtomIndices=AtomIndices,GetRandomConformations=False,NumConfsToGet=None,Which=None,Stride=Stride,SkipKCenters=False,DiscardFirstN=0,DiscardLastN=0)
        
        return Gens

    def EvenSampling(self, NumGens):
        print "doing even sampling"

        startStates = {}
        i = 0
        while i < NumGens:
            startStates[i] = 1
            i +=1

        return startStates

    def AdaptiveSampling(self, C, NumSims):
        mat = np.matrix(C, dtype="float64")
        nStates = mat.shape[0]

        calc = ssaCalculator( 1, mat, 1.0/nStates, evalList=[1], nNewSamples=NumSims)
        calc.displayContributions( bling=True )

        # write output
        startStateList = {}
        nOut = 0
        for i in range(len(calc.varianceContributions[0][:,9] )):
            if calc.varianceContributions[0][i,9] > 0:
                startStateList[i] = round(calc.varianceContributions[0][i,9])
                nOut += startStateList[i]

        # make sure get desired number of sims
        if nOut < NumSims:
            nHave = nOut
            nOut = 0
            for i in startStateList.keys():
                startStateList[i] = round(float(NumSims)/nHave*startStateList[i])
                nOut += startStateList[i]
            i = 0
            while nOut < NumSims:
                startStateList[startStateList.keys()[i]] += 1
                nOut += 1
                i += 1
                if i >= len(startStateList):
                    i = 0

        return startStateList

def CreateCopernicusProject(ConfFilename, FileList):
    # setup reference conformation
    if ConfFilename == None:
        print "No reference conf! ERROR"
        return
    Conf=Conformation.Conformation.LoadFromPDB(ConfFilename)

    NumTraj = 0
    PartsLenList = []
    for TrajPartList in FileList:
        PartsLenList.append(NumTraj)
        n = len(TrajPartList)
        NumTraj += n
    PartsLenList = np.array(PartsLenList, "int32")

    if NumTraj==0:
        print("No data found!  ERROR")
        return

    S=dict()
    S["FileList"] = FileList
    S["NumTrajs"]=NumTraj
    S["PartsLenList"] = PartsLenList
    S["TrajFilePath"]=None
    S["TrajFileBaseName"]=None
    S["TrajFileType"]=None
    S["ConfFilename"]=ConfFilename
    S["TrajLengths"]=None #we set this below
    P1 = CopernicusProject(S)
    P1["NumTrajs"]=NumTraj
    P1["PartsLenList"] = PartsLenList

    try:
        os.mkdir("./Data")
    except OSError:
        pass

    LenList=[]
    for i in range(NumTraj):
        f=P1.GetTrajFilename(i)
        print f
        LenList.append(Trajectory.Trajectory.LoadTrajectoryFile(f,Conf=Conf,JustInspect=True)[0])

    P1["TrajLengths"]=np.array(LenList)

    return P1

