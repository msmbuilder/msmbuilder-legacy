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
import numpy as np
import scipy.io
from msmbuilder import tpt
from msmbuilder import arglib
logger = logging.getLogger('msmbuilder.scripts.CalculateMFPTs')


parser = arglib.ArgumentParser(description="""
Calculates the mean first passage times (MFPTs) to one or all states.
Returns: MFPTs_X.dat or PairwiseMFPTs.dat, where X is the state ID.

Note: PairwiseMFPTs.dat is written if state=-1.  Otherwise, MFPTs_X.dat
is written.
""")
parser.add_argument('tProb')
parser.add_argument(
    'state', help='''ID of state to which we calculate MFPT.  If state=-1, then calculate all pairwise MFPTs.''', default="-1", type=int)
parser.add_argument('output_dir', default='.')


def run(T, state):
    if state != -1:
        logger.info("Calculating MFPTs to state %d" % state)
        m = tpt.calculate_mfpt([state], T)
        logger.info("Finished calculating MFPTs to state %d" % state)
    else:
        logger.info("Calculating MFPTs to all states")
        m = tpt.calculate_all_to_all_mfpt(T)
        logger.info("Finished calculating MFPTs to all states")

    return m


def entry_point():
    args = parser.parse_args()

    T = scipy.io.mmread(args.tProb)
    state = int(args.state)
    print(args.state, state)

    # Check output isn't taken
    if state == -1:
        base_filename = "PairwiseMFPTs.dat"
    else:
        base_filename = "MFPTs_%d.dat" % state

    output_filename = os.path.join(args.output_dir, base_filename)
    arglib.die_if_path_exists(output_filename)

    MFPTs = run(T, state)

    np.savetxt(output_filename, MFPTs)
    logger.info("Saved output to %s" % output_filename)


if __name__ == "__main__":
    entry_point()
