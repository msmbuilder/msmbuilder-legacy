# This script is part of MSMBuilder
# 
# Call this shell script to generate a reference MSM with a working
# version of MSMBuilder2. You should have already been provided with
# a tested reference... this tool is mostly to aid this process for
# the developers
#
# -- TJL <tjlane@stanford.edu> 8/30/11

import os
import shutil
from ReferenceParameters import *

print("Building reference MSM...")

# set up reference directory
try:
    shutil.rmtree(ReferenceDir)
except:
    pass

os.mkdir(ReferenceDir)
os.chdir(ReferenceDir)

print("Created and moved to dir: reference")

os.system("ConvertDataToHDF.py -s %s/dipeptide.pdb -I %s/XTC/"%(TutorialDir,TutorialDir))
os.system("CreateAtomIndices.py -s %s/dipeptide.pdb"%TutorialDir)
os.system("Cluster.py -u %d -k %d -r %s -m %d"%(Stride,NumClusters,RMSDCutoff,LocalKMedoids))
os.system("Assign.py")
os.system("CalculateImpliedTimescales.py -l %d,%d -X %d -e %d"%(MinLagtime,MaxLagtime,LagtimeInterval,NumEigen))
os.system("BuildMSM.py -l %d"%Lagtime)
os.system("GetRandomConfs.py -c %d"%NumRandomConformations)
os.system("CalculateRMSD.py -s %s/dipeptide.pdb -I Data/Gens.lh5"%TutorialDir)
os.system("CalculateClusterRadii.py -a Data/Assignments.h5")
os.system("CalculateProjectRMSD.py -s %s/dipeptide.pdb"%TutorialDir)
os.system("DoTPT.py -F %s/F_test.dat -U %s/U_test.dat -T Data/tProb.mtx "%(TutorialDir,TutorialDir))
os.system("SavePDBs.py -H %s -c %d"%(StatesToSaveString,NumRandomConformations))
os.system("PCCA.py -M %d -T Data/tProb.mtx"%NumMacroStates)


