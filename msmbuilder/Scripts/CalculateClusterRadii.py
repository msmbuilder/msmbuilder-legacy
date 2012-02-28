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

import ArgLib

from msmbuilder import Serializer

def run(args):
    assignFn, rmsdFn, MinState, MaxState = args
    Assignments = Serializer.LoadData(assignFn)
    Assignments = Assignments.flatten()
    RMSDs = Serializer.LoadData(rmsdFn)
    RMSDs = RMSDs.flatten()

    # Check Output
    if os.path.exists("ClusterRadii.dat"):
        print "Error: 'ClusterRadii.dat' already exists! Exiting."
        sys.exit(1)

    NumStates=Assignments.max()+1
    radii = numpy.zeros(NumStates)

    for i in range(MinState, MaxState):
        print "Calculating radius of state:", i
        snapshots = numpy.where(Assignments == i)[0]
        if len(snapshots) > 0: radii[i] = RMSDs[snapshots].mean()
        else: radii[i] = -1; print "WARNING: No assignments found for cluster:", i

    return radii

if __name__ == "__main__":
    print """
\nCalculates the cluster radius for all clusters in the model. Here, we define
radius is simply the average RMSD of all conformations in a cluster to its
generator. Does this by taking averaging the distance of each assigned state to
its generator.

Output: A flat txt file, 'ClusterRadii.dat', the average RMSD distance to the
generator in nm.\n"""

    arglist=["assignments", "assrmsd", "procs", "output"]
    options=ArgLib.parse(arglist, Custom=[("assignments", "Path to assignments file. Default: Data/Assignments.Fixed.h5", "Data/Assignments.Fixed.h5")])
    print sys.argv

    if options.output == 'NoOutputSet': output = "ClusterRadii.dat"
    else: output = options.output

    # Check Output
    if os.path.exists(output):
        print "Error:", output, "already exists! Exiting."
        sys.exit(1)

    nProc = int(options.procs)
    print "Running on %d processors" % nProc
    Assignments = Serializer.LoadData(options.assignments)

    MinStates = []
    MaxStates = []
    NumStates=Assignments.max()+1
    StateRange = NumStates/nProc
    for k in range(nProc):
        print "Partition:", k, "Range:", k*StateRange, (k+1)*StateRange-1
        MinStates.append( k*StateRange )
        MaxStates.append( (k+1)*StateRange-1 )
    MaxStates[-1] = NumStates #Ensure that we go to the end

    pool = multiprocessing.Pool(processes=nProc)
    input = zip(nProc*[options.assignments], nProc*[options.assrmsd], MinStates, MaxStates)
    result = pool.map_async(run, input)
    result.wait()
    radii = result.get()

    print radii
    radii = numpy.array(radii).sum(axis=0)
    assert len(radii) == NumStates
    numpy.savetxt(output, radii)
    print "Wrote:", output
