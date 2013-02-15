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

import os
import numpy as np
import scipy.io
from msmbuilder import arglib
import msmbuilder.io
from msmbuilder import MSMLib
import logging
logger = logging.getLogger('msmbuilder.scripts.BuildMSM')

def get_num_assignments(assignments_list):
    
    flat_assignments = np.concatenate([assignments.flatten() 
                                       for assignments in assignments_list])

    return len(flat_assignments[np.where(flat_assignments != -1)])

def run(lag_time, assignments_list, symmetrize='MLE', input_mapping="None", 
        out_dir="./Data/"):

    # set the filenames for output
    tProb_fn = os.path.join(out_dir, "tProb.mtx")
    tCounts_fn = os.path.join(out_dir, "tCounts.mtx")
    map_fn = os.path.join(out_dir, "Mapping.dat")
    pops_fn = os.path.join(out_dir, "Populations.dat")
    if len(assignments_list) == 1:
        assignments_fn_list = [os.path.join(out_dir, "Assignments.Fixed.h5")]
    else:
        assignments_fn_list = [os.path.join(out_dir, 
                                            "Assignments.Fixed.%d.h5" % i)
                               for i in xrange(len(assignments_list))]


    # make sure none are taken
    output_list = [tProb_fn, tCounts_fn, map_fn, pops_fn] + assignments_fn_list
    arglib.die_if_path_exists(output_list)

    # if given, apply mapping to assignments
    for i in xrange(len(assignments_list)):
        if input_mapping != "None":
            MSMLib.apply_mapping_to_assignments(assignments_list[i], 
                                                input_mapping)

    n_assigns_before_trim = get_num_assignments(assignments_list)

    num_states = np.unique(np.concatenate([ np.unique(ass[np.where(ass != -1)]) 
                                           for ass in assignments_list])).shape[0]

    counts = MSMLib.get_count_matrix_from_assignments(assignments_list[0], 
                                                      n_states=num_states,
                                                      lag_time=lag_time, 
                                                      sliding_window=True)

    for i in xrange(1, len(assignments_list)):
        print i
        counts = counts + \
                 MSMLib.get_count_matrix_from_assignments(assignments_list[i],
                                                          n_states=num_states,
                                                          lag_time=lag_time,
                                                          sliding_window=True)

    rev_counts, t_matrix, populations, mapping = \
        MSMLib.build_msm(counts, symmetrize=symmetrize, ergodic_trimming=True)

    for i in xrange(len(assignments_list)):
        MSMLib.apply_mapping_to_assignments(assignments_list[i], mapping)

    n_assigns_after_trim = get_num_assignments(assignments_list)

    # if had input mapping, then update it
    if input_mapping != "None":
        mapping = mapping[input_mapping]

    # Print a statement showing how much data was discarded in trimming
    percent = (1.0 - float(n_assigns_after_trim) / 
                     float(n_assigns_before_trim)) * 100.0
    logger.warning("Ergodic trimming discarded: "
                   "%f percent of your data", percent)

    # Save all output
    scipy.io.mmwrite(tProb_fn, t_matrix)
    scipy.io.mmwrite(tCounts_fn, rev_counts)
    np.savetxt(map_fn, mapping, "%d")
    np.savetxt(pops_fn, populations)
    for i in xrange(len(assignments_fn_list)):
        assignments_fn = assignments_fn_list[i]
        assignments = assignments_list[i]
        msmbuilder.io.saveh(assignments_fn, assignments)

    for output in output_list:
        logger.info("Wrote: %s", output)

    return

if __name__ == "__main__":
    parser = arglib.ArgumentParser(description=
        """Estimates the counts and transition matrices from an
        Assignments.h5 file. Reversible models can be calculated either from 
        naive symmetrization or estimation of the most likely reversible 
        matrices (MLE, recommended). Also calculates the equilibrium 
        populations for the model produced. Outputs will be saved in the 
        directory of your choice.
        \nOutput: tCounts.mtx, tProb.mtx, Populations.dat,  Mapping.dat,
        Assignments.Fixed.h5, tCounts.UnSym.mtx""")
    parser.add_argument('assignments', nargs='+', help="""Assignments file
        produced by MSMBuilder. You can input multiple assignments files. In
        this case, a single model will be constructed using all trajectories.
        Additionally, there will be a 'fixed' assignment file for each input
        assignments file numbered by the order they were input.""")
    parser.add_argument('symmetrize', help="""Method by which to estimate a
        symmetric counts matrix. Symmetrization ensures reversibility, but may 
        skew dynamics. We recommend maximum likelihood estimation (MLE) when 
        tractable, else try Transpose. It is strongly recommended you read the
        documentation surrounding this choice.""", default='MLE',
        choices=['MLE', 'Transpose', 'None'])
    parser.add_argument('lag_time', help='''Lag time to use in model (in
        number of snapshots. EG, if you have snapshots every 200ps, and set the
        lagtime=50, you'll get a model with a lagtime of 10ns)''', type=int)
    parser.add_argument('mapping', help='''Mapping, EG from microstates to 
        macrostates. If given, this mapping will be applied to the specified 
        assignments before creating an MSM.''', default="None")
    parser.add_argument('output_dir')
    args = parser.parse_args()

    assignments_list = []
    for i in xrange(len(args.assignments)):
        try:
            assignments = msmbuilder.io.loadh(args.assignments[i], 'arr_0')
        except KeyError:
            assignments = msmbuilder.io.loadh(args.assignments[i], 'Data')
        assignments_list.append(assignments)
        
    if args.mapping != "None":
        args.mapping = np.array(np.loadtxt(args.mapping), dtype=int)

    run(args.lag_time, assignments_list, args.symmetrize, args.mapping, 
        args.output_dir)
