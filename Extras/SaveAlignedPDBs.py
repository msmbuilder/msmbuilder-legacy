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

import os, sys

from msmbuilder import arglib
from msmbuilder import Trajectory
from msmbuilder.metric_LPRMSD import LPRMSD, LPTraj, ReadPermFile
from collections import defaultdict
import numpy as np
import random

def run(project, assignments, conformations_per_state, states, output_dir, gens_file, atom_indices, permute_indices, alt_indices):
    if states == "all":
        states = np.arange(assignments.max()+1)
    # This is a dictionary: {generator : ((traj1, frame1), (traj1, frame3), (traj2, frame1), ... )}
    inverse_assignments = defaultdict(lambda: [])
    for i in xrange(assignments.shape[0]):
        for j in xrange(assignments.shape[1]):
            inverse_assignments[assignments[i,j]].append((i,j))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print "Setting up the metric."
    rmsd_metric = LPRMSD(atom_indices,permute_indices,alt_indices)
    # Create a trajectory of generators and prepare it.
    gens_traj = Trajectory.LoadTrajectoryFile(gens_file)
    p_gens_traj = rmsd_metric.prepare_trajectory(gens_traj)
    # This trickery allows us to get the correct number of leading
    # zeros in the output file name no matter how many generators we have
    digits = len(str(max(states)))
    formstr = '\"Cluster-%%0%ii.pdb\"' % digits
    # Loop through the generators.
    for s in states:
        if len(inverse_assignments[s]) == 0:
            raise ValueError('No assignments to state! %s' % s)
        if conformations_per_state == 'all':
            confs = inverse_assignments[s]
        else:
            random.shuffle(inverse_assignments[s])
            if len(inverse_assignments[s]) >= conformations_per_state:
                confs = inverse_assignments[s][0:conformations_per_state]
            else:
                confs = inverse_assignments[s]
                print 'Not enough assignments in state %s' % s
        FrameDict = {}
        for (traj, frame) in confs:
            FrameDict.setdefault(traj,[]).append(frame)
        # Create a single trajectory corresponding to the frames that
        # belong to the current generator.
        cluster_traj = project.GetEmptyTrajectory()
        TrajNums = set([i[0] for i in confs])
        for i in TrajNums:
            T = project.LoadTraj(i)[np.array(FrameDict[i])]
            cluster_traj += T
        print "Loaded %i conformations for generator %i, now aligning." % (len(cluster_traj), s)
        # Prepare the trajectory, align to the generator, and reassign the coordinates.
        p_cluster_traj = rmsd_metric.prepare_trajectory(cluster_traj)
        rmsd, xout = rmsd_metric.one_to_all_aligned(p_gens_traj, p_cluster_traj, s)
        p_cluster_traj['XYZList'] = xout
        outfnm = eval(formstr) % s
        # Now save the PDB file.
        print "Done aligning; saving to %s" % os.path.join(output_dir,outfnm)
        p_cluster_traj.SaveToPDB(os.path.join(output_dir,outfnm))
                
if __name__ == '__main__':
    parser = arglib.ArgumentParser(description="""
Pulls the specified number of random structures (or optionally all
structures) from each state in an assignments file, aligned to the
generators. Specify which states to pull from with space-seperated
ints

Output: A bunch of PDB files named: State<StateIndex>-<Conformation>, inside
the directory 'PDBs'
Note: If you want to get structures for all states, it is more efficient
to use GetRandomConfs.py""")
    parser.add_argument('project')
    parser.add_argument('assignments', default='Data/Assignments.Fixed.h5')
    parser.add_argument('conformations_per_state', default=5, type=int,
        description='Number of conformations to sample from each state: to specify ALL of the conformations, pass the integer -1.')
    parser.add_argument('states', nargs='+', type=int,
        description='''Which states to sample from. Pass a list of integers, separated
        by whitespace. To specify ALL of the states (Although the script GetRandomConfs.py
        is more efficient for this purpose), pass the integer -1.''')

    parser.add_argument('lprmsd_atom_indices', description='Regular atom indices', default='AtomIndices.dat')
    parser.add_argument('lprmsd_permute_atoms', default='None', description='''Atom labels to be permuted.
    Sets of indistinguishable atoms that can be permuted to minimize the RMSD. On disk this should be stored as
    a list of newline separated indices with a "--" separating the sets of indices if there are
    more than one set of indistinguishable atoms''')
    parser.add_argument('lprmsd_alt_indices', description='Alternate atom indices', default='AltIndices.dat')

    parser.add_argument('generators', description='''Trajectory file containing
    the structures of each of the cluster centers.  Produced using Cluster.py.''', default='Data/Gens.lh5')


    parser.add_argument('output_dir', default='PDBs')
    args = parser.parse_args()
    
    if -1 in args.states:
        print "Ripping PDBs for all states"
        args.states = 'all'

    if args.conformations_per_state == -1:
        print "Getting all PDBs for each state"
        args.conformations_per_state = 'all'

    atom_indices = np.loadtxt(args.lprmsd_atom_indices, np.int)
    
    if args.lprmsd_permute_atoms == 'None':
        permute_indices = None
    else:
        permute_indices = ReadPermFile(args.lprmsd_permute_atoms)

    if args.lprmsd_alt_indices == 'None':
        alt_indices = None
    else:
        alt_indices = np.loadtxt(args.lprmsd_alt_indices, np.int)

    run(args.project, args.assignments['Data'], args.conformations_per_state,
         args.states, args.output_dir, args.generators, atom_indices, permute_indices, alt_indices)

