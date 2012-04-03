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
#

import os
import sys
import multiprocessing
import numpy

from Emsmbuilder import arglib
from Emsmbuilder import Serializer

def run(args):
    assignFn, rmsdFn, MinState, MaxState = args
    Assignments = Serializer.LoadData(assignFn)
    Assignments = Assignments.flatten()
    RMSDs = Serializer.LoadData(rmsdFn)
    RMSDs = RMSDs.flatten()

    NumStates = Assignments.max() + 1
    radii = numpy.zeros(NumStates)

    for i in range(MinState, MaxState):
        print "Calculating radius of state:", i
        snapshots = numpy.where(Assignments == i)[0]
        if len(snapshots) > 0: radii[i] = RMSDs[snapshots].mean()
        else: radii[i] = -1; print "WARNING: No assignments found for cluster:", i

    return radii

if __name__ == "__main__":
    
    parser = arglib.ArgumentParser(description="""
Calculates the cluster radius for all clusters in the model. Here, we define
radius is simply the average RMSD of all conformations in a cluster to its
generator. Does this by taking averaging the distance of each assigned state to
its generator.

Note that this script is not yet equipped to handle different distance metrics
(other than RMSD)

Output: A flat txt file, 'ClusterRadii.dat', the average RMSD distance to the
generator in nm.""")
    parser.add_argument('assignments', type=str, default='Data/Assignments.Fixed.h5')
    parser.add_argument('assignments_rmsd', description='''Path to assignment
        RMSD file.''', default='Data/Assignments.h5.RMSD')
    parser.add_argument('procs', description="""Number of physical processors/cores
        to use""", default=1, type=int)
    parser.add_argument('output', default='ClusterRadii.dat')
    args = parser.parse_args()
    arglib.die_if_path_exists(args.output)
    
    
    MinStates = []
    MaxStates = []
    NumStates = Serializer.LoadData(args.assignments).max() + 1
    StateRange = NumStates / args.procs
    for k in range(args.procs):
        print "Partition:", k, "Range:", k * StateRange, (k + 1) * StateRange - 1
        MinStates.append(k*StateRange)
        MaxStates.append((k + 1) * StateRange - 1)
    MaxStates[-1] = NumStates #Ensure that we go to the end

    pool = multiprocessing.Pool(processes=args.procs)
    input = zip(args.procs*[args.assignments], args.procs*[args.assignments_rmsd], MinStates, MaxStates)
    result = pool.map_async(run, input)
    result.wait()
    radii = result.get()

    #print radii
    radii = numpy.array(radii).sum(axis=0)
    assert len(radii) == NumStates

    numpy.savetxt(args.output, radii)
    print "Wrote:", args.output
