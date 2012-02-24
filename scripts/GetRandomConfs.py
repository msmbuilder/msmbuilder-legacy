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

from Emsmbuilder import Project
from Emsmbuilder import Serializer

import ArgLib

def run(P1, Assignments, NumConfsPerState, output):

    # Check output paths
    if os.path.exists(output):
        print "Error: File %s already exists! Exiting." % output
        sys.exit(1)

    NumStates=max(Assignments.flatten())+1
    print "Pulling", NumConfsPerState, "for each of", NumStates
    RandomConfs=P1.GetRandomConfsFromEachState(Assignments,NumStates,NumConfsPerState)
    RandomConfs.SaveToLHDF(output)
    RandomConfs.SaveToPDB(output+".pdb")
    print "Wrote output to:", output

    return


if __name__ == "__main__":
    print """
\nPulls a certain number of random conformations from each cluster. Returns these
as an HDF5 file that contains one long chain of these conformations that looks
like a Trajectory. If you selected to sample N conformations from each cluster,
the first N conformations are from cluster 0, the next N from cluster 1, etc.\n"""

    print "Output default: XRandomConfs.lh5, where X=Number of Conformations.\n"
    arglist=["projectfn", "assignments", "conformations", "output"]
    options=ArgLib.parse(arglist, Custom=[
        ("output","The name of the RandomConfs trajectory (.lh5) to write. XRandomConfs.lh5, where X=Number of Conformations.", "NoOutputSet"),
        ("assignments", "Path to assignments file. Default: Data/Assignments.Fixed.h5", "Data/Assignments.Fixed.h5")])
    if options.output=='NoOutputSet': output='%sRandomConfs.lh5' % options.conformations
    else: output=options.output
    print sys.argv

    P1=Project.Project.LoadFromHDF(options.projectfn)
    Assignments=Serializer.LoadData(options.assignments)
    NumConfsPerState=int(options.conformations)

    run(P1, Assignments, NumConfsPerState, output)
