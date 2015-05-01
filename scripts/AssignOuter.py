#!/usr/bin/env python
# This file is part of MSMBuilder.
#
# Copyright 2014 Stanford University
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


#===============================================================================
# Imports
#===============================================================================

import logging

from mdtraj import io

from msmbuilder import arglib, assigning
from msmbuilder.metrics import solvent

#===============================================================================
# Globals
#===============================================================================

logger = logging.getLogger('msmbuilder.scripts.AssignOuter')
parser = arglib.ArgumentParser(description=
    """Combine the results of two clusterings. Specifically, if a conformation
    is in state i after clustering with metric 1, and it is in state j
    after clustering with metric 2, we assign it to a new state (ij)

    This is a naive way of combining two metrics to give a singular
    assignment of states.""")
parser.add_argument('assignment1', default='./Data1/Assignments.h5',
                    help='First assignment file')
parser.add_argument('assignment2', default='./Data2/Assignments.h5',
                    help='Second assignment file')
parser.add_argument('assignment_out', default='OuterProductAssignments.h5',
                    help='Output file')


def run(assign1_fn, assign2_fn, out_fn):
    assign1 = io.loadh(assign1_fn, 'arr_0')
    assign2 = io.loadh(assign2_fn, 'arr_0')

    new_assignments = assigning.outer_product_assignment(assign1, assign2)
    io.saveh(out_fn, new_assignments)
    logger.info('Saved outer product assignments to %s', out_fn)


def entry_point():
    args = parser.parse_args()
    arglib.die_if_path_exists(args.assignment_out)

    run(args.assignment1, args.assignment2, args.assignment_out)


if __name__ == "__main__":
    entry_point()
