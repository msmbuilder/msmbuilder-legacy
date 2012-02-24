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

from Emsmbuilder import MSMLib
from Emsmbuilder import Serializer

import ArgLib

def run(MinLagtime, MaxLagtime, Interval, NumEigen, AssignmentsFn, symmetrize, nProc, output):

    # Check output isn't taken
    if os.path.exists(output):
        print "Error: File %s already exists! Exiting." % output
        sys.exit(1)

    # Setup some model parameters
    Assignments=Serializer.LoadData(AssignmentsFn)
    NumStates=max(Assignments.flatten())+1
    if NumStates <= NumEigen-1: 
        NumEigen=NumStates-2
        print "Number of requested eigenvalues exceeds the rank of the transition matrix! Defaulting to the maximum possible number of eigenvalues."
    del Assignments

    print "Getting %d eigenvalues (timescales) for each lagtime..." % NumEigen
    lagTimes = range(MinLagtime, MaxLagtime+1, Interval)
    print "Building MSMs at the following lag times:",  lagTimes

    # Get the implied timescales (eigenvalues)
    impTimes = MSMLib.GetImpliedTimescales(AssignmentsFn, NumStates,
                                           lagTimes, NumImpliedTimes=NumEigen,
                                           Slide=True, Symmetrize=symmetrize, nProc=nProc)
    numpy.savetxt(output, impTimes)
    return


if __name__ == "__main__":
    print """
\nCalculates the implied timescales of a set of assigned data, up to
the argument 'lagtime'. Returns: ImpliedTimescales.dat, a flat file that
contains all the lag times.\n"""

    arglist=["assignments", "lagtime", "interval", "eigvals", "symmetrize", "procs", "output"]
    options=ArgLib.parse(arglist, Custom=[
        ("lagtime", "The lagtime range to calculate. Pass two floats as X,Y with NO WHITESPACE, where X is the lowest timescale you want and Y is the biggest. EG: '-l 5,50'.", None),
        ("output", "The name of the  implied timescales data file (use .dat extension)", "ImpliedTimescales.dat") ])
    print sys.argv

    LagTimes = options.lagtime.split(',')
    MinLagtime = int(LagTimes[0])
    MaxLagtime = int(LagTimes[1])

    # Pass the symmetric flag
    if options.symmetrize in ["None", "none", None]: symmetrize = None
    else: symmetrize = options.symmetrize

    run(MinLagtime, MaxLagtime, int(options.interval), int(options.eigvals), options.assignments, symmetrize, int(options.procs), options.output)
