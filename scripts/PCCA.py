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
import logging
import scipy.io
import numpy as np
from mdtraj import io
from msmbuilder import MSMLib
from msmbuilder import lumping
from msmbuilder import arglib
logger = logging.getLogger('msmbuilder.scripts.PCCA')


float_or_none = lambda s: None if s.lower() == 'none' else float(s)
parser = arglib.ArgumentParser(description="""
Applies the (f)PCCA(+) algorithm to lump your microstates into macrostates. You may
specify a transition matrix if you wish - this matrix is used to determine the
dynamics of the microstate model for lumping into kinetically relevant
macrostates.

Output: MacroAssignments.h5, a new assignments HDF file, for the Macro MSM.""")
parser.add_argument('assignments', default='Data/Assignments.Fixed.h5')
parser.add_argument('num_macrostates', type=int)
parser.add_argument('tProb')
parser.add_argument('output_dir')
parser.add_argument('algorithm', help='Which algorithm to use',
                    choices=['PCCA', 'PCCA+'], default='PCCA')
parser.add_argument_group('Extra PCCA+ Options')
parser.add_argument('flux_cutoff', help='''Discard eigenvectors below
    this flux''', default='None', type=float_or_none)
parser.add_argument('objective_function', help='''Minimize which PCCA+
    objective function (crisp_metastability, metastability, or crispness)''',
                    default="crisp_metastability")
parser.add_argument(
    'do_minimization', help='Use PCCA+ minimization', default=True)


def run_pcca(num_macrostates, assignments, tProb):
    logger.info("Running PCCA...")
    if len(np.unique(assignments[np.where(assignments != -1)])) != tProb.shape[0]:
        raise ValueError(
            'Different number of states in assignments and tProb!')
    PCCA = lumping.PCCA(tProb, num_macrostates)
    MAP = PCCA.microstate_mapping

    # MAP the new assignments and save, make sure don't
    # mess up negaitve one's (ie where don't have data)
    MSMLib.apply_mapping_to_assignments(assignments, MAP)

    return MAP, assignments


def run_pcca_plus(num_macrostates, assignments, tProb, flux_cutoff=0.0,
                  objective_function="crispness", do_minimization=True):

    logger.info("Running PCCA+...")
    pcca_plus = lumping.PCCAPlus(
        tProb, num_macrostates, flux_cutoff=flux_cutoff,
        do_minimization=do_minimization, objective_function=objective_function)

    A, chi, MAP = pcca_plus.A, pcca_plus.chi, pcca_plus.microstate_mapping

    MSMLib.apply_mapping_to_assignments(assignments, MAP)

    return chi, A, MAP, assignments

def entry_point():
    args = parser.parse_args()

    # load args
    try:
        assignments = io.loadh(args.assignments, 'arr_0')
    except KeyError:
        assignments = io.loadh(args.assignments, 'Data')

    tProb = scipy.io.mmread(args.tProb)

    # workaround for arglib funniness?
    if args.do_minimization in ["False", "0"]:
        args.do_minimization = False
    else:
        args.do_minimization = True

    if args.algorithm == 'PCCA':
        MacroAssignmentsFn = os.path.join(
            args.output_dir, "MacroAssignments.h5")
        MacroMapFn = os.path.join(args.output_dir, "MacroMapping.dat")
        arglib.die_if_path_exists([MacroAssignmentsFn, MacroMapFn])

        MAP, assignments = run_pcca(args.num_macrostates, assignments, tProb)

        np.savetxt(MacroMapFn, MAP, "%d")
        io.saveh(MacroAssignmentsFn, assignments)
        logger.info("Saved output to: %s, %s", MacroAssignmentsFn, MacroMapFn)

    elif args.algorithm == 'PCCA+':
        MacroAssignmentsFn = os.path.join(
            args.output_dir, "MacroAssignments.h5")
        MacroMapFn = os.path.join(args.output_dir, "MacroMapping.dat")
        ChiFn = os.path.join(args.output_dir, 'Chi.dat')
        AFn = os.path.join(args.output_dir, 'A.dat')

        arglib.die_if_path_exists([MacroAssignmentsFn, MacroMapFn, ChiFn, AFn])

        chi, A, MAP, assignments = run_pcca_plus(args.num_macrostates,
                                                 assignments, tProb, args.flux_cutoff, objective_function=args.objective_function,
                                                 do_minimization=args.do_minimization)

        np.savetxt(ChiFn, chi)
        np.savetxt(AFn, A)
        np.savetxt(MacroMapFn, MAP, "%d")
        io.saveh(MacroAssignmentsFn, assignments)
        logger.info('Saved output to: %s, %s, %s, %s',
                    ChiFn, AFn, MacroMapFn, MacroAssignmentsFn)
    else:
        raise Exception()


if __name__ == "__main__":
    entry_point()
