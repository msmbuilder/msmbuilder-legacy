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
import os
import numpy

import ArgLib

from msmbuilder import Project
from msmbuilder import Trajectory
from msmbuilder import Serializer

def run(Project, atomindices, Generators, OutDir, CheckpointFn):

    AssFilename     = OutDir + "/Assignments.h5"
    AssRMSDFilename = OutDir + "/Assignments.h5.RMSD"
    ArgLib.CheckPath(AssFilename)
    ArgLib.CheckPath(AssRMSDFilename)

    # Check to see if we're going to be reading from a checkpoint file
    if CheckpointFn == None:         # If not, check output isn't occupied
        Assignments, RMSD, WhichTrajsWereAssigned = Project.AssignProject(Generators, AtomIndices=atomindices)
        
    else:                           # If so, set some paths and check they exist
        OldAssign = CheckpointFn
        OldRMSD   = CheckpointFn + '.RMSD'
        if not os.path.isfile(OldAssign):
            print "\nDidn't find a checkpoint file: %s" % OldAssign
            print "Checkpointing data there during assignment...\n"
            Assignments, RMSD, WhichTrajsWereAssigned = Project.AssignProject(Generators, AtomIndices=atomindices, CheckpointFile=CheckpointFn)
        else:
            if not os.path.isfile(OldRMSD):
                print "Error: Cannot find file needed to load checkpoint: %s" % OldRMSD
                print "Forcing exit."
                sys.exit(1)
            print "Found valid checkpoint files: %s, %s. Proceeding from there."
            Assignments, RMSD, WhichTrajsWereAssigned = Project.AssignProject(Generators, AtomIndices=atomindices, AssFilename=OldAssign, AssRMSDFilename=OldRMSD)
    
    Serializer.SaveData(AssFilename, Assignments)
    Serializer.SaveData(AssRMSDFilename, RMSD)
    print "Wrote: %s, %s"%(AssFilename,AssRMSDFilename)
    return


if __name__ == "__main__":
    print """\nAssigns data to generators that was not originally used in the
clustering.\n
Output:
-- Assignments.h5: a matrix of assignments where each row is a vector
corresponding to a data trajectory. The values of this vector are the cluster
assignments.
-- Assignments.h5.RMSD: Gives the RMSD from the assigned frame to its Generator.

Checkpointing:
    Assign.py supports checkpoint files. To generate a checkpoint file, specify a
    new path with the --checkpoint (-k) flag. To restart from an old checkpoint,
    point the --checkpoint flag to an old checkpoint file.
    
    NOTE: The checkpoints actually consist of *two* files, one with an extra .RMSD
    appended to the filename. BOTH are necessary for restart. \n"""

    arglist=["projectfn", "generators", "atomindices", "outdir", "checkpoint"]
    options=ArgLib.parse(arglist)
    print sys.argv

    P1=Project.Project.LoadFromHDF(options.projectfn)
    AInd=numpy.loadtxt(options.atomindices, int)
    Generators=Trajectory.Trajectory.LoadTrajectoryFile(options.generators, Conf=P1.Conf)

    if options.checkpoint == 'None':
        checkpoint = None
    else:
        checkpoint = options.checkpoint
    
    run(P1, AInd, Generators, options.outdir, checkpoint)
