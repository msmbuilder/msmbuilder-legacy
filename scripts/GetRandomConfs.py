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

import numpy as np
from msmbuilder import Trajectory
from msmbuilder import Project
from msmbuilder import MSMLib
from msmbuilder import arglib
from msmbuilder import io
import logging
logger = logging.getLogger(__name__)

def run(project, assignments, num_confs_per_state, random_source=None):
    """
    Pull random confs from each state in an MSM
    
    Parameters
    ----------
    project : msmbuilder.Project
        Used to load up the trajectories, get topology
    assignments : np.ndarray, dtype=int
        State membership for each frame
    num_confs_per_state : int
        number of conformations to pull from each state
    random_source : numpy.random.RandomState, optional
        If supplied, random numbers will be pulled from this random source,
        instead of the default, which is np.random. This argument is used
        for testing, to ensure that the random number generator always
        gives the same stream.
        
    Notes
    -----
    A new random_source can be initialized by calling numpy.random.RandomState(seed)
    with whatever seed you like. See http://stackoverflow.com/questions/5836335/consistenly-create-same-random-numpy-array
    for some discussion.
                
    """
    
    if random_source is None:
        random_source = np.random
    
    n_states = max(assignments.flatten()) + 1
    logger.info("Pulling %s confs for each of %s confs", num_confs_per_state, n_states)
    
    inv = MSMLib.invert_assignments(assignments)
    xyzlist = []
    for s in xrange(n_states):
        trj, frame = inv[s]
        # trj and frame are a list of indices, such that
        # project.load_traj(trj[i])[frame[i]] is a frame assigned to state s
        for j in xrange(num_confs_per_state):
            r = random_source.randint(len(trj))
            xyz = Trajectory.read_frame(project.traj_filename(trj[r]), frame[r])
            xyzlist.append(xyz)
            
    # xyzlist is now a list of (n_atoms, 3) arrays, and we're going
    # to stack it along the third dimension 
    xyzlist = np.dstack(xyzlist)
    # load up the conf to get the topology, put then pop in the new coordinates
    output = project.load_conf()
    output['XYZList'] = xyzlist
    
    return output
    


if __name__ == "__main__":
    parser = arglib.ArgumentParser(description="""
Pulls a certain number of random conformations from each cluster. Returns these
as an HDF5/PDB/XTC file that contains one long chain of these conformations that looks
like a Trajectory. If you selected to sample N conformations from each cluster,
the first N conformations are from cluster 0, the next N from cluster 1, etc.

Output default: XRandomConfs.lh5, where X=Number of Conformations.""")
    parser.add_argument('project')
    parser.add_argument('assignments', default='Data/Assignments.Fixed.h5')
    parser.add_argument('output', help="""The name of the RandomConfs
        trajectory (.lh5) to write. XRandomConfs.lh5, where X=Number of
        Conformations.""", default='XRandomConfs')
    parser.add_argument('conformations_per_state', help='''Number of
        conformations to randomly sample from your data per state''', type=int)
    parser.add_argument('format', help='''Format to output the data in. Note
        that the PDB format is uncompressed and not efficient. For XTC, you can view
        the trajectory using your project's topology file''', default='lh5',
        choices=['pdb', 'xtc', 'lh5'])    
    args = parser.parse_args()
    
    if args.output == 'XRandomConfs':
            args.output = '%dRandomConfs.%s' % (args.conformations_per_state, args.format)
    
    try:
        assignments = io.loadh(args.assignments, 'arr_0')
    except KeyError:
       assignments = io.loadh(args.assignments, 'Data')
    project = Project.load_from(args.project)
    
    random_confs = run(project, assignments, args.conformations_per_state)
        
    if args.format == 'pdb':
        random_confs.SaveToPDB(args.output)
    elif args.format == 'lh5':
        random_confs.SaveToLHDF(args.output)
    elif args.format == 'xtc':
        random_confs.SaveToXTC(args.output)
    else:
        raise ValueError('Unrecognized format')

    logger.info("Saved output to %s", args.output)
