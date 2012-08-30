# These should be set manually to match the reference
# These choices were made to roughly match the tutorial, except without k-medoids (which is stochastic).
import os

DeleteWhenFinished=True

this_file_dir = os.path.abspath(os.path.dirname(__file__))
ReferenceDir = os.path.join(this_file_dir, "UnitTestReference/")
WorkingDir   = os.path.join(this_file_dir, "UnitTestWorkingDir/")
TutorialDir   = os.path.join(this_file_dir, "../Tutorial/")

PDBFn         = TutorialDir+"/native.pdb"
ProjectFn     = "ProjectInfo.h5"
GensPath      = "Data/Gens.lh5"

print "Set working dir: %s"   % WorkingDir
print "Set reference dir: %s" % ReferenceDir

### MSM Settings ###

Symmetrize             = "MLE"

Stride                 = 1
NumClusters            = 1000000
RMSDCutoff             = 0.03
LocalKMedoids          = 0
GlobalKMedoids         = 0

MinLagtime             = 1
MaxLagtime             = 25
LagtimeInterval        = 1
NumEigen               = 3

Lagtime                = 5

NumRandomConformations = 3
StatesToSaveString     = "3 5"
NumMacroStates         = 5

#MinState and MaxState specify which states to calculate cluster radii for
MinState               = 0
MaxState               = 74
#Note this HAS to match the number of states, otherwise ClusterRadii unittest will fail.
