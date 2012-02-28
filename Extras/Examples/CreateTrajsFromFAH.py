import os
from scipy import array

from msmbuilder import Project, CreateMergedTrajectoriesFromFAH




ProjectFilename="ProjectInfo.h5"
PDBFilename="native.pdb"
DataDir="../10424/PROJ10424/"
NumRuns=50000
NumClones=1

#You can use an array of Runs, clones to specify which trajectories to include / exclude.  The code below includes only Runs 45000, ... , 49999 and Clone 0.  
WhichRunsClones=array([[i, 0] for i in range(45000,50000)]).astype('int')

P1=CreateMergedTrajectoriesFromFAH.CreateMergedTrajectoriesFromFAH(PDBFilename,DataDir,NumRuns,NumClones,OutFileType=".lh5",MaxGen=1000,MinGen=25,DiscardFirstN=1,WhichRunsClones=WhichRunsClones,Usetrjcat=False,ProjectFilename=ProjectFilename)
