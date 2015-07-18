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

# TJL 2011, PANDE GROUP

import numpy as np
import logging
import scipy.io
from mdtraj import io
from msmbuilder import tpt
from msmbuilder import arglib
logger = logging.getLogger('msmbuilder.scripts.FindPaths')


parser = arglib.ArgumentParser(description=
"""Finds the highest flux paths through an MSM.
Returns: an HDF5 file (default: Paths.h5), which contains three items:
(1) The highest flux pathways (a list of ints)
(2) The bottlenecks in these pathways (a list of 2-tuples)
(3) The flux of each pathway""")

parser.add_argument('number', help='''Number of pathways you want
    to retreive''', type=int)
parser.add_argument('tprob', help='Transition probability matrix',
                    default='tProb.mtx')
parser.add_argument('starting', help='''Vector of states in the
    starting/reactants/unfolded ensemble.''', default='U_states.dat')
parser.add_argument('ending', help='''Vector of states in the
    ending/products/folded ensemble.''', default='F_states.dat')
parser.add_argument('output', default='Paths.h5')


def run(tprob, sources, sinks, num_paths):

    net_flux = tpt.calculate_net_fluxes(sources, sinks, tprob)

    paths, fluxes = tpt.get_paths(sources, sinks, net_flux, num_paths=num_paths)

    # We have to pad the paths with -1s to make a square array
    max_length = np.max([len(p) for p in paths])

    padded_paths = -1 * np.ones((len(paths), max_length))
    for i, path in enumerate(paths):
        padded_paths[i, :len(path)] = np.array(path)

    return padded_paths, np.array(fluxes)


def entry_point():
    args = parser.parse_args()

    sinks = np.loadtxt(args.ending).astype(int).reshape((-1,))
    sources = np.loadtxt(args.starting).astype(int).reshape((-1,))
    # .reshape((-1,)) ensures that a single number turns into an array with a shape

    tprob = scipy.io.mmread(args.tprob)

    arglib.die_if_path_exists(args.output)
    paths, fluxes = run(tprob, sources, sinks, args.number)

    io.saveh(args.output, paths=paths, fluxes=fluxes)
    logger.info('Saved output to %s', args.output)

if __name__ == "__main__":
    entry_point()
