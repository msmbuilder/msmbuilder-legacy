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

import sys, os
import numpy as np
import scipy.io

from msmbuilder import arglib
import logging
from msmbuilder import Serializer, MSMLib
logger = logging.getLogger(__name__)

pjoin = lambda a,b : os.path.join(a, b)

if __name__ == "__main__":
    parser = arglib.ArgumentParser("""Build a rate matrix MSM.

Note: this uses a lag_time of 1 to get the transition counts, and uses
rate estimators that use the *dwell_times*.

The *correct* likelihood function to use for estimating the rate matrix when
the data is sampled at a discrete frequency is open for debate. This
likelihood function doesn't take into account the error in the lifetime estimates
from the discrete sampling. Other methods are currently under development.
    
Output: tCounts.mtx, K.mtx, Populations.dat,  Mapping.dat,
Assignments.Fixed.h5, tCounts.UnSym.mtx""")

    parser.add_argument('assignments')
    parser.add_argument('symmetrize', choices=['none', 'transpose', 'mle'])
    parser.add_argument('outdir')
    args = parser.parse_args()
    assignments = Serializer.LoadData(args.assignments)
    
    
    ratemtx_fn = pjoin(args.outdir, 'K.mtx')
    tcounts_fn = pjoin(args.outdir, 'tCounts.mtx')
    unsym_fn = pjoin(args.outdir, 'tCounts.UnSym.mtx')
    mapping_fn = pjoin(args.outdir, 'Mapping.dat')
    fixed_fn = pjoin(args.outdir, 'Assignments.Fixed.h5')
    pops_fn = pjoin(args.outdir, 'Populations.dat')
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    outlist = [ratemtx_fn, tcounts_fn, unsym_fn, fixed_fn, pops_fn]
    for e in outlist:
        arglib.die_if_path_exists(e)
    
    # if lag time is not one, there's going to be a unit mispatch between
    # what you get and what you're expecting. 
    lag_time = 1
    counts, rev_counts, t_matrix, populations, mapping = MSMLib.build_msm(assignments,
        lag_time=lag_time, symmetrize=args.symmetrize)
    K = MSMLib.estimate_rate_matrix(rev_counts, assignments)
    
    np.savetxt(pops_fn, populations)
    np.savetxt(mapping_fn, mapping, "%d")
    scipy.io.mmwrite(ratemtx_fn, K)
    scipy.io.mmwrite(tcounts_fn, rev_counts)
    scipy.io.mmwrite(unsym_fn, counts)
    Serializer.SaveData(fixed_fn, assignments)
    
    for e in outlist:
        logger.info('Saved %s' % e)
