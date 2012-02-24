from numpy import *

from msmbuilder.CopernicusProject import *


ReferenceConf = "native.pdb"
FileList = [["Trajectories/trj0.xtc"], ["Trajectories/trj1.xtc"]]

# indices to use in clustering, numbering from 0
AtomIndices = arange(576)

Proj = CreateCopernicusProject(ReferenceConf, FileList)
Generators = Proj.ClusterProject(AtomIndices=AtomIndices)
Assignments = Proj.AssignProject(Generators, AtomIndices=AtomIndices)
NumGens = Generators["XYZList"].shape[0]
StartStates = Proj.EvenSampling(NumGens)

# assignments in "Data" element of dictionary as numpy array
print Assignments["Data"].shape

# dictinoary of states to start simulations from with weight for that state
for key in StartStates.keys():
    print "will start %d sims in state %d" % (StartStates[key], key)


