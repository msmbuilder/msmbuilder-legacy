import os
from msmbuilder import Project,CreateMergedTrajectoriesFromFAH

ProjectFilename="ProjectInfo.h5"
PDBFilename="native.pdb"

NumGen=3
NumTraj=1

if not os.path.exists("./Trajectories/"):
        FList=[["xtc/frame%d.xtc"%i for i in range(NumGen)] for j in range(NumTraj)]
        CreateMergedTrajectoriesFromFAH.CreateMergedTrajectories("./native.pdb",FList)


if not os.path.exists(ProjectFilename):
        P1=Project.CreateProjectFromDir(Filename=ProjectFilename,ConfFilename=PDBFilename,TrajFileType=".lh5")
