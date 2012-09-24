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

import sys
import os
import numpy

from msmbuilder import MSMLib
from msmbuilder import io
from msmbuilder import arglib
from msmbuilder import msm_analysis

import logging
logger = logging.getLogger(__name__)


def run(MinLagtime, MaxLagtime, Interval, NumEigen, AssignmentsFn, symmetrize, nProc, output):

    arglib.die_if_path_exists(output)
    
    # Setup some model parameters
    try:
        Assignments = io.loadh(AssignmentsFn, 'arr_0')
    except KeyError:
        Assignments = io.loadh(AssignmentsFn, 'Data')
    
    NumStates = max(Assignments.flatten())+1
    if NumStates <= NumEigen-1: 
        NumEigen = NumStates-2
        logger.warning("Number of requested eigenvalues exceeds the rank of the transition matrix! Defaulting to the maximum possible number of eigenvalues.")
    del Assignments

    logger.info("Getting %d eigenvalues (timescales) for each lagtime...", NumEigen)
    lagTimes = range(MinLagtime, MaxLagtime+1, Interval)
    logger.info("Building MSMs at the following lag times: %s", lagTimes)

    # Get the implied timescales (eigenvalues)
    impTimes = msm_analysis.get_implied_timescales(AssignmentsFn, lagTimes,
        n_implied_times=NumEigen, sliding_window=True, symmetrize=symmetrize,
        n_procs=nProc)
    numpy.savetxt(output, impTimes)
    return


if __name__ == "__main__":
    parser = arglib.ArgumentParser(description="""
\nCalculates the implied timescales of a set of assigned data, up to
the argument 'lagtime'. Returns: ImpliedTimescales.dat, a flat file that
contains all the lag times.\n""")
    parser.add_argument('assignments', type=str)
    parser.add_argument('lagtime', help="""The lagtime range to calculate.
        Pass two ints as X,Y with NO WHITESPACE, where X is the lowest
        timescale you want and Y is the biggest. EG: '-l 5,50'.""")
    parser.add_argument('output', help="""The name of the  implied
        timescales data file (use .dat extension)""", default='ImpliedTimescales.dat')
    parser.add_argument('procs', help='''Number of concurrent processes
        (cores) to use''', default=1, type=int)
    parser.add_argument('eigvals', help="""'Number of eigenvalues
        (implied timescales) to retrieve at each lag time""", default=10, type=int)
    parser.add_argument('interval', help="""Number of times (intervals)
        to calculate lagtimes for""", default=20, type=int)
    parser.add_argument('symmetrize', help="""Method by which to estimate a
        symmetric counts matrix. Symmetrization ensures reversibility, but may skew
        dynamics. We recommend maximum likelihood estimation (MLE) when tractable,
        else try Transpose. It is strongly recommended you read the documentation
        surrounding this choice.""", default='MLE',
        choices=['MLE', 'Transpose', 'None'])
    args = parser.parse_args()

    LagTimes = args.lagtime.split(',')
    MinLagtime = int(LagTimes[0])
    MaxLagtime = int(LagTimes[1])

    # Pass the symmetric flag
    if args.symmetrize in ["None", "none", None]:
        args.symmetrize = None

    run(MinLagtime, MaxLagtime, args.interval, args.eigvals, args.assignments,
        args.symmetrize, args.procs, args.output)
