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
from msmbuilder import Trajectory, io, Project
from lprmsd import LPRMSD, ReadPermFile
from collections import defaultdict
from msmbuilder.clustering import concatenate_trajectories
import copy
import numpy as np
import random
import resource

def get_size(start_path = '.'):
    total_size = 0
    num_files = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
            num_files += 1
    return num_files, total_size

def get_file_size(fnm):
    return os.path.getsize(fnm)

def run(project, assignments, conformations_per_state, states, output_dir, gens_file, atom_indices, permute_indices, alt_indices, total_memory):
    if states == "all":
        states = np.arange(assignments['arr_0'].max()+1)
    # This is a dictionary: {generator : ((traj1, frame1), (traj1, frame3), (traj2, frame1), ... )}
    inverse_assignments = defaultdict(lambda: [])
    for i in xrange(assignments['arr_0'].shape[0]):
        for j in xrange(assignments['arr_0'].shape[1]):
            inverse_assignments[assignments['arr_0'][i,j]].append((i,j))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print "Setting up the metric."
    rmsd_metric = LPRMSD(atom_indices,permute_indices,alt_indices)
    # This trickery allows us to get the correct number of leading
    # zeros in the output file name no matter how many generators we have
    digits = len(str(max(states)))
    # Create a trajectory of generators and prepare it.
    if os.path.exists(gens_file):
        gens_traj = Trajectory.load_trajectory_file(gens_file)
        p_gens_traj = rmsd_metric.prepare_trajectory(gens_traj)
        formstr_pdb = '\"Generator-%%0%ii.pdb\"' % digits
    
    formstr_xtc = '\"Cluster-%%0%ii.xtc\"' % digits
    print "Loading up the trajectories."
    traj_bytes = sum([get_file_size(project.traj_filename(i)) for i in range(project.n_trajs)])
    LoadAll = 0
    MaxMem = 0.0
    # LPW This is my hack that decides whether to load trajectories into memory, or to read them from disk.
    if (traj_bytes * 5) < total_memory * 1073741824: # It looks like the Python script uses roughly 5x the HDF file size in terms of memory.
        print "Loading all trajectories into memory."
        LoadAll = 1
        AllTraj = [project.load_traj(i) for i in np.arange(project.n_trajs)]
        #print "After loading trajectories, memory usage is % .3f GB" % (float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1048576)
    
    if not os.path.exists(gens_file):
        if not 'AllTraj' in locals():
            raise Exception(('To get away with not supplying a Gens.lh5 structure to align to for each state '
                             'you need to have enough memory to load all the trajectories simultaniously. This could be worked around...'))
        print 'Randomly Sampling from state for structure to align everything to'
        centers_list = []
        for s in states:
            chosen = inverse_assignments[np.random.randint(len(inverse_assignments[s]))]
            centers_list.append(AllTraj[chosen[0]][chosen[1]])
        gens_traj = concatenate_trajectories(centers_list)
        p_gens_traj = rmsd_metric.prepare_trajectory(gens_traj)
        formstr_pdb = '\"Center-%%0%ii.pdb\"' % digits
        
    
    cluster_traj = project.empty_traj()
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
        if "XYZList" in cluster_traj:
            cluster_traj["XYZList"] = None
            #cluster_traj.pop("XYZList")
        print "Generator %i" % s,
        TrajNums = set([i[0] for i in confs])
        for i in TrajNums:
            if LoadAll:
                T = AllTraj[i][np.array(FrameDict[i])]
            else:
                T = project.load_traj(i)[np.array(FrameDict[i])]
            cluster_traj += T
        print " loaded %i conformations, aligning" % len(cluster_traj),
        # Prepare the trajectory, align to the generator, and reassign the coordinates.
        p_cluster_traj = rmsd_metric.prepare_trajectory(cluster_traj)
        rmsd, xout = rmsd_metric.one_to_all_aligned(p_gens_traj, p_cluster_traj, s)
        p_cluster_traj['XYZList'] = xout.copy()
        # Now save the generator / cluster to a PDB / XTC file.
        outpdb = eval(formstr_pdb) % s
        outxtc = eval(formstr_xtc) % s
        this_gen_traj = p_gens_traj[s]
        print ", saving PDB to %s" % os.path.join(output_dir,outpdb),
        this_gen_traj.save_to_pdb(os.path.join(output_dir,outpdb))
        print ", saving XTC to %s" % os.path.join(output_dir,outxtc),
        p_cluster_traj.save_to_xtc(os.path.join(output_dir,outxtc))
        print ", saved"
        NowMem = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1048576
        if NowMem > MaxMem:
            MaxMem = NowMem
    #print "This script used at least % .3f GB of memory" % MaxMem
                
if __name__ == '__main__':
    parser = arglib.ArgumentParser("""
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
        help='Number of conformations to sample from each state: to specify ALL of the conformations, pass the integer -1.')
    parser.add_argument('states', nargs='+', type=int,
        help='''Which states to sample from. Pass a list of integers, separated
        by whitespace. To specify ALL of the states (Although the script GetRandomConfs.py
        is more efficient for this purpose), pass the integer -1.''')

    parser.add_argument('lprmsd_atom_indices', help='Regular atom indices', default='AtomIndices.dat')
    parser.add_argument('lprmsd_alt_indices', default='None', help='Alternate atom indices')
    parser.add_argument('lprmsd_permute_atoms', default='None', help='''Atom labels to be permuted.
    Sets of indistinguishable atoms that can be permuted to minimize the RMSD. On disk this should be stored as
    a list of newline separated indices with a "--" separating the sets of indices if there are
    more than one set of indistinguishable atoms''')
    parser.add_argument('total_memory_gb', default=4, type=int, help='Available memory in GB; this determines whether to load all trajectories into memory or to read them one-by-one from disk.')

    parser.add_argument('generators', help='''Trajectory file containing
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
    assignments = io.loadh(args.assignments)
    project = Project.load_from(args.project)
    
    if args.lprmsd_permute_atoms == 'None':
        permute_indices = None
    else:
        permute_indices = ReadPermFile(args.lprmsd_permute_atoms)

    if args.lprmsd_alt_indices == 'None':
        alt_indices = None
    else:
        alt_indices = np.loadtxt(args.lprmsd_alt_indices, np.int)

    run(project, assignments, args.conformations_per_state,
         args.states, args.output_dir, args.generators, atom_indices, permute_indices, alt_indices, args.total_memory_gb)

