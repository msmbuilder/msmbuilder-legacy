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

#TJL 2011, PANDE GROUP

import sys
import numpy as np
import os
import scipy.io

from msmbuilder import tpt
import msmbuilder.io 
from msmbuilder import arglib
import logging
logger = logging.getLogger('msmbuilder.scripts.FindPaths')

def run(tprob, sources, sinks, num_paths=np.inf, flux_cutoff=1-1E-10):

    net_flux = tpt.calculate_net_fluxes(sources, sinks, tprob)

    paths, fluxes = tpt.get_paths(sources, sinks, net_flux, 
        num_paths=num_paths, flux_cutoff=flux_cutoff)

    return paths, fluxes


if __name__ == "__main__":
    parser = arglib.ArgumentParser(description=
"""Finds the highest flux paths through an MSM.
Returns: an HDF5 file (default: Paths.h5), which contains three items:
(1) The highest flux pathways (a list of ints)
(2) The bottlenecks in these pathways (a list of 2-tuples)
(3) The flux of each pathway

Paths.h5 can be read by RenderPaths.py which generates a .dot file capturing these paths.""")
    
    parser.add_argument('num_paths', help='''Number of pathways you want
        to retrieve''', type=int, default=np.inf)
    parser.add_argument('flux_cutoff', type=float, default=1, 
        help='''find paths until the percentage of explained flux is 
        greater than this cutoff (between 0 and 1)''')
    parser.add_argument('tprob', help='Transition probability matrix',
        default='tProb.mtx')
    parser.add_argument('starting', help='''Vector of states in the
        starting/reactants/unfolded ensemble.''', default='U_states.dat')
    parser.add_argument('ending', help='''Vector of states in the
        ending/products/folded ensemble.''', default='F_states.dat')
    parser.add_argument('output', default='Paths.h5')
    args = parser.parse_args()
    
    arglib.die_if_path_exists(args.output)

    sinks = np.loadtxt(args.ending).astype(int)
    sources = np.loadtxt(args.starting).astype(int)
    tprob = scipy.io.mmread(args.tprob)
    
    # deal with case where have single start or end state
    # TJL note: this should be taken care of in library now... keeping it just in case
    if sinks.shape == ():
        sinks = np.array([sinks])
    if sources.shape == ():
        sources = np.array([sources])
    
    paths, fluxes = run(tprob, sources, sinks, args.num_paths, args.flux_cutoff)
    
    msmbuilder.io.saveh(args.output, paths=paths, fluxes=fluxes)
    logger.info('Saved output to %s', args.output)
