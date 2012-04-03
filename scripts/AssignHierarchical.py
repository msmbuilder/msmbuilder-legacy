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
from Emsmbuilder import Serializer
from Emsmbuilder.clustering import Hierarchical
from Emsmbuilder import arglib

if __name__ == "__main__":
    parser = arglib.ArgumentParser(description='Assign data using a hierarchical clustering')
    parser.add_argument('hierarichal_clustering_zmatrix', default='ZMatrix.h5', 
        description='Path to hierarchical clustering zmatrix', type=Hierarchical.load_from_disk)
    parser.add_argument('num_states', description='Number of States', default='none')
    parser.add_argument('cutoff_distance', description='Maximum cophenetic distance', default='none')
    parser.add_argument('assignments', type=str)
    args = parser.parse_args()
    
    k = args.num_states if args.num_states != 'none' else None
    d = args.cutoff_distance if args.num_states != 'none' else None
    if k is None and d is None:
        print >> sys.stderr, '%s: Error: You need to supply either a number of states or a cutoff distance' % (os.path.split(sys.argv[0])[1])
        sys.exit(1)
    
    assignments = args.hierarichal_clustering_zmatrix.get_assignments(k=k, cutoff_distance=d)
    
    Serializer.SaveData(args.assignments, assignments)
