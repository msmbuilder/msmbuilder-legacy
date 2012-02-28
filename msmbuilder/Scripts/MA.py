#!/usr/bin/python
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

import sys
import numpy

from msmbuilder import Serializer
from msmbuilder import Project
from msmbuilder import Trajectory

import ArgLib

def run(partition, project, atomindices, generators,NumProcs):

    WhichTrajs=numpy.arange(P1["NumTrajs"])
    #Select the subset of trajectories that need to be assigned by this ProcID
    WhichTrajs=WhichTrajs[partition::NumProcs]
    
    Assignments,RMSD,WhichTrajsWereAssigned=project.AssignProject(generators, AtomIndices=atomindices,WhichTrajs=WhichTrajs)

    Serializer.SaveData("Assignments.%d.Ass" % partition,Assignments)
    Serializer.SaveData("Assignments.%d.RMSD" % partition,RMSD)
    Serializer.SaveData("Assignments.%d.WhichTrajs" % partition, WhichTrajsWereAssigned)
    print "Saved Assignments.%d.h5, Assignments.%d.h5.RMSD" % (partition, partition)
    return

if __name__ == "__main__":
    print "\nMA.py is a MultiAssign working script called by AssignOnPBS.py. IT SHOULD RARELY BE CALLED INDEPENDENTLY. Look into using Assign.py or AssignOnPBS.py to assign your data.\n\nThe final output is: \nAssignments.h5: a matrix of assignments where each row is a vector corresponding to a data trajectory. The values of this vector are the cluster assignments.\nAssignments.h5.RMSD: Gives the RMSD from the assigned frame to its Generator.\n\nNote: Look into doing a parallel assignment with AssignOnCertainty.py for faster results!\n"

    arglist=["projectfn", "generators", "atomindices", "procs", "procid"]
    options=ArgLib.parse(arglist)
    print sys.argv

    partition=int(options.procid)
    NumProcs=int(options.procs)
    P1=Project.Project.LoadFromHDF(options.projectfn)
    AInd=numpy.loadtxt(options.atomindices, int)
    Generators=Trajectory.Trajectory.LoadTrajectoryFile(options.generators,Conf=P1.Conf)

    run(partition, P1, AInd, Generators,NumProcs)

    
