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
import numpy as np

from msmbuilder import MSMLib
from msmbuilder import io
from msmbuilder import arglib
from msmbuilder import msm_analysis

import logging
logger = logging.getLogger(__name__)


def run(min_lag_time, max_lag_time, interval, num_eigen, assignments_list, 
        symmetrize, num_procs, output):

    arglib.die_if_path_exists(output)
    
    # Setup some model parameters
    
    flat_assignments = np.concatenate([assignments.flatten()
                                       for assignments in assignments_list])
    num_states = len(np.unique(flat_assignments[np.where(flat_assignments != -1)]))

    if num_states <= num_eigen-1: 
        num_eigen = num_states-2
        logger.warning("Number of requested eigenvalues exceeds the rank of the "
                       "transition matrix! Defaulting to the maximum possible "
                       "number of eigenvalues.")

    logger.info("Getting %d eigenvalues (timescales) for each lagtime...", 
                num_eigen)

    lag_times = range(min_lag_time, max_lag_time+1, interval)
    logger.info("Building MSMs at the following lag times: %s", lag_times)

    # Get the implied timescales (eigenvalues)
    imp_times = msm_analysis.get_implied_timescales(assignments_list, lag_times,
        num_implied_times=num_eigen, sliding_window=True, symmetrize=symmetrize,
        num_procs=num_procs)

    np.savetxt(output, imp_times)

    return


if __name__ == "__main__":
    parser = arglib.ArgumentParser(description="""
\nCalculates the implied timescales of a set of assigned data, up to
the argument 'lagtime'. Returns: ImpliedTimescales.dat, a flat file that
contains all the lag times.\n""")
    parser.add_argument('assignments', nargs='+', help="""Assignments file(s) 
        constructed by msmbuilder.""")
    parser.add_argument('lag_times', nargs=2, type=int, help="""The lagtime range 
        to calculate. Pass two ints as "X Y" with whitespace in between, where X is
        the lowest timescale you want and Y is the biggest. EG: '-l 5 50'.""")
    parser.add_argument('output', help="""The name of the  implied
        timescales data file (use .dat extension)""", 
        default='ImpliedTimescales.dat')
    parser.add_argument('procs', help='''Number of concurrent processes
        (cores) to use''', default=1, type=int)
    parser.add_argument('eigvals', help="""'Number of eigenvalues
        (implied timescales) to retrieve at each lag time""", default=10, type=int)
    parser.add_argument('interval', help="""Calculate implied timescales between
        the lag time range at the specified interval (frames).""", default=20, 
        type=int)
    parser.add_argument('symmetrize', help="""Method by which to estimate a
        symmetric counts matrix. Symmetrization ensures reversibility, but may skew
        dynamics. We recommend maximum likelihood estimation (MLE) when tractable,
        else try Transpose. It is strongly recommended you read the documentation
        surrounding this choice.""", default='MLE',
        choices=['MLE', 'Transpose', 'None'])
    args = parser.parse_args()

    min_lag_time, max_lag_time = args.lag_times

    # Pass the symmetric flag
    if args.symmetrize in ["None", "none", None]:
        args.symmetrize = None

    # Load the assignments
    assignments_list = []
    for i in xrange(len(args.assignments)):
        try:
            assignments = io.loadh(args.assignments[i], 'arr_0')
        except KeyError:
            assignments = io.loadh(args.assignments[i], 'Data')
        assignments_list.append(assignments)
        
    run(min_lag_time, max_lag_time, args.interval, args.eigvals, assignments_list,
        args.symmetrize, args.procs, args.output)
