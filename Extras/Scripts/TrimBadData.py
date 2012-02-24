#!/usr/bin/env python
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

import ArgLib
import sys
from os import path
from numpy import array, argmax, sum, where
from scipy import io
from msmbuilder import MSMLib, Serializer, Trajectory

def run(Generators, Assignments, AssignmentsRMSD, LagTime, MaxRMSD):

    # Check output paths
    if path.exists("Assignments.Fixed.h5"):
        print "Error: File 'Assignments.Fixed.h5' already exists! Exiting."
        sys.exit(1)
    if path.exists("tCounts.Fixed.mtx"):
        print "Error: File 'tCounts.Fixed.mtx5' already exists! Exiting."
        sys.exit(1)
    if path.exists("Gens.Fixed.lh5"):
        print "Error: File 'Gens.Fixed.lh5' already exists! Exiting."
        sys.exit(1)

    NumStates=max(Assignments.flatten())+1

    # Trim Data that is futher than a certain value from the cluster center
    print "Trimming by RMSD..."
    PrevMinusOne = where(Assignments == -1)
    MSMLib.TrimHighRMSDToCenters(Assignments, AssignmentsRMSD, MaxRMSD)
    if sum(Assignments.flatten()+1) == 0:
        print "No assignments left after trimming! Exiting. Try again with less stringent trimming or recluster."
        sys.exit()

    # Construct the smallest lag-time MSM possible and look for disjoint data
    print "Trimming non-ergodic components & keeping largest..."
    Counts=MSMLib.GetCountMatrixFromAssignments(Assignments, NumStates, LagTime=LagTime, Slide=True)
    C, MAP = MSMLib.ErgodicTrim(Counts)

    io.mmwrite("tCounts.Fixed.mtx", C)
    print "Wrote: tCounts.Fixed.mtx"
 
    A = Assignments
    MinusOne = where(A == -1)
    kept=float(len(MinusOne[0]) - len(PrevMinusOne[0])) / float(len(Assignments.flatten())) * 100.
    print "Discarding: %f percent of assigned data." % kept

    Assignments = MAP[A]
    Assignments[MinusOne] = -1
    Serializer.SaveData("Assignments.Fixed.h5",Assignments)
    print "Wrote: Assignments.Fixed.h5"

    Generators["XYZList"] = Generators["XYZList"][where( MAP!=-1 )[0]]
    Generators.SaveToLHDF("Gens.Fixed.lh5")
    print "Wrote: Gens.Fixed.lh5"
    
    return

if __name__ == "__main__":
    print "\nPerforms two operations that allow for an MSM face-lift:\n(1) Trims assignments that are futher than a specified RMSD value from their generator.\n(2) Finds the maximally populated state, and discards any data that is not connected to the graph containing that state.\nOutput: (1) \nA new Assignments HDF5: Assignments.Fixed.h5\nA new Generators lh5: Gens.Fixed.lh5\n(3) A Counts Matrix: tCounts.Fixed.mtx"

    arglist=["generators", "assignments", "rmsd", "lagtime"]
    options=ArgLib.parse(arglist)
    print sys.argv

    Gens=Trajectory.Trajectory.LoadFromLHDF(options.generators)
    Assignments=Serializer.LoadData(options.assignments)
    RMSD=Serializer.LoadData(options.assignments+".RMSD")

    run(Gens, Assignments, RMSD, int(options.lagtime), float(options.rmsd))
