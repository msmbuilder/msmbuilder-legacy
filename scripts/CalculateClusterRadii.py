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

##############################################################################
# Imports
##############################################################################

import logging
import numpy as np
from mdtraj import io
from msmbuilder import arglib
from msmbuilder.MSMLib import invert_assignments

##############################################################################
# Globals
##############################################################################

logger = logging.getLogger('msmbuilder.scripts.CalculateClusterRadii')
parser = arglib.ArgumentParser(description="""
Calculates the cluster radius for all clusters in the model. Here, we define
radius is simply the average distance of all conformations in a cluster to its
generator. Does this by taking averaging the distance of each assigned state to
its generator.

Output: A flat txt file, 'ClusterRadii.dat', the average RMSD distance to the
generator, measured by what ever distance metric was used in assigning.""")

parser.add_argument(
    'assignments', type=str, default='Data/Assignments.Fixed.h5')
parser.add_argument('distances', help='''Path to assignment
    distances file.''', default='Data/Assignments.h5.distances')
parser.add_argument('output', default='ClusterRadii.dat')

##############################################################################
# Code
##############################################################################


def main(assignments, distances):
    """
    Calculate the mean radii of each state
    
    Parameters
    ----------
    assignments : np.ndarray, shape=[n_trajs, n_frames], dtype=int
        The array of assignments, indicating which cluster each frame is
        assigned to.
    distances : np.ndarray, shape=[n_trajs, n_frames], dtype=float/double
        The array of distances, indicating the distance (measured by some metric
        used in the Assign step) from each frame to its assigned generator
    
    Returns
    -------
    radii : np.ndarray, shape=[n_states], dtype=float/double
        The mean radii of each state
    """

    inverse_mapping = invert_assignments(assignments)
    states = inverse_mapping.keys()

    # don't count the minus one state, since it indicates the absense of
    # an assignments
    if -1 in states:
        states.remove(-1)
    n_states = len(states)

    radii = np.nan * np.ones(n_states)
    for s in states:
        trj, frame = inverse_mapping[s]
        radii[s] = distances[trj, frame].mean()

    return radii


if __name__ == "__main__":
    args = parser.parse_args()
    arglib.die_if_path_exists(args.output)

    try:
        assignments = io.loadh(args.assignments, 'arr_0')
        distances = io.loadh(args.distances, 'arr_0')
    except KeyError:
        assignments = io.loadh(args.assignments, 'Data')
        distances = io.loadh(args.distances, 'Data')

    radii = main(assignments, distances)

    np.savetxt(args.output, radii)
    logger.info("Wrote: %s", args.output)
