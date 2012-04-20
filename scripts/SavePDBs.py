#!/usr/bin/python
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

import os, sys

from msmbuilder import arglib
from collections import defaultdict
import numpy as np
import random

def run(project, assignments, conformations_per_state, states, output_dir):
    if states == "all":
        states = np.arange(assignments.max()+1)
    
    inverse_assignments = defaultdict(lambda: [])
    for i in xrange(assignments.shape[0]):
        for j in xrange(assignments.shape[1]):
            inverse_assignments[assignments[i,j]].append((i,j))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    empty_traj = project.GetEmptyTrajectory()
    for s in states:
        if len(inverse_assignments[s]) == 0:
            raise ValueError('No assignments to state! %s' % s)
        
        random.shuffle(inverse_assignments[s])
        if len(inverse_assignments[s]) >= conformations_per_state:
            confs = inverse_assignments[s][0:conformations_per_state]
        else:
            confs = inverse_assignments[s]
            print 'Not enough assignments in state %s' % s
        
        for i, (traj, frame) in enumerate(confs):
            outfile = os.path.join(output_dir, 'State%d-%d.pdb' % (s, i))
            if not os.path.exists(outfile):
                print 'Saving state %d (traj %d, frame %d) as %s' % (s, traj, frame, outfile)
                xyz = project.ReadFrame(traj, frame)
                empty_traj['XYZList'] = np.array([xyz])
                empty_traj.SaveToPDB(outfile)
            else:
                print 'Skipping %s. Already exists' % outfile
                
if __name__ == '__main__':
    parser = arglib.ArgumentParser(description="""
Pulls the specified number of random structures from each state in an assignments
file. Specify which states to pull from from with space-seperated ints

Output: A bunch of PDB files named: State<StateIndex>-<Conformation>, inside
the directory 'PDBs'
Note: If you want to get structures for all states, it is more efficient
to use GetRandomConfs.py""")
    parser.add_argument('project')
    parser.add_argument('assignments', default='Data/Assignments.Fixed.h5')
    parser.add_argument('conformations_per_state', default=5, type=int)
    parser.add_argument('states', nargs='+')
    parser.add_argument('output_dir', default='PDBs')
    args = parser.parse_args()
    if any([a in args.states for a in ['All', 'all', 'ALL']]):
        print "Ripping PDBs for all states"
        args.states = 'all'
    else:
        try:
            args.states = [int(s) for s in args.states]
        except:
            print >> sys.stderr, 'Error: Could not parse states as integers'
            sys.exit(1)
        args.states = np.unique(args.states)
    
    run(args.project, args.assignments['Data'], args.conformations_per_state,
         args.states, args.output_dir)

