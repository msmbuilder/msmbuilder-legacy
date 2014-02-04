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
import numpy as np
from mdtraj import io
from msmbuilder import arglib
import logging
logger = logging.getLogger('msmbuilder.scripts.TrimAssignments')

parser = arglib.ArgumentParser(description="""
Trims assignments based on the distance to their generator. Useful for
eliminating bad assignments from a coase clustering. Note that this
discards (expensive!) data, so should only be used if an optimal
clustering is not available.

Note: Check your cluster sized with CalculateClusterRadii.py to get
a handle on how big they are before you trim. Recall the radius is the
*average* distance to the generator, here you are enforcing the
*maximum* distance.

Output: A trimmed assignments file (Assignments.Trimmed.h5).""")
parser.add_argument('assignments', default='Data/Assignments.Fixed.h5')
parser.add_argument('distances', default='Data/Assignments.h5.distances')
parser.add_argument('rmsd_cutoff', help="""distance value at which to trim,
    in. Data further than this value to its generator will be
    discarded. Note: this is measured with whatever distance metric you used to cluster""", type=float)
parser.add_argument('output', default='Data/Assignments.Trimmed.h5')


def run(assignments, distances, cutoff):
    number = np.count_nonzero(distances > cutoff)
    logger.info('Discarding %d assignments', number)

    assignments[np.where(distances > cutoff)] = -1
    return assignments


if __name__ == "__main__":
    args = parser.parse_args()

    arglib.die_if_path_exists(args.output)

    try:
        assignments = io.loadh(args.assignments, 'arr_0')
        distances = io.loadh(args.distances, 'arr_0')
    except KeyError:
        assignments = io.loadh(args.assignments, 'Data')
        distances = io.loadh(args.distances, 'Data')

    trimmed = run(assignments, distances, args.rmsd_cutoff)

    io.saveh(args.output, trimmed)
    logger.info('Saved output to %s', args.output)
