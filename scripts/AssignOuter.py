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


from msmbuilder import arglib, io
from msmbuilder.metrics import solvent
import logging
logger = logging.getLogger('msmbuilder.scripts.AssignOuter')

parser = arglib.ArgumentParser(description='''Assign data to the outer product
                                              space of two preliminary
                                              assignments''')
parser.add_argument('assignment1', default='./Data1/Assignments.h5',
                    help='First assignment file')
parser.add_argument('assignment2', default='./Data2/Assignments.h5',
                    help='Second assignment file')
parser.add_argument('assignment_out', default='OuterProductAssignments.h5',
                    help='Output file')


def main(ass1_fn, ass2_fn):
    hierarchical = Hierarchical.load_from_disk(zmatrix_fn)
    assignments = hierarchical.get_assignments(k=k, cutoff_distance=d)
    return assignments

if __name__ == "__main__":
    args = parser.parse_args()
    arglib.die_if_path_exists(args.assignment_out)

    opa = solvent.OuterProductAssignment(args.assignment1, args.assignment2)
    new_assignments = opa.get_product_assignments()
    io.saveh(args.assignment_out, new_assignments)
    logger.info('Saved outer product assignments to %s', args.assignment_out)
