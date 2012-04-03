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

import numpy as np
from Emsmbuilder import Trajectory
from Emsmbuilder.metrics import RMSD
from Emsmbuilder import arglib

def run(project, pdb, traj_fn, atom_indices):

    #project = Project.LoadFromHDF(options.projectfn)
    traj = Trajectory.LoadTrajectoryFile(traj_fn,Conf=project.Conf)
    
    # you could replace this with your own metric if you like
    metric = RMSD(atom_indices)

    ppdb = metric.prepare_trajectory(pdb)
    ptraj = metric.prepare_trajectory(traj)
    distances = metric.one_to_all(ppdb, ptraj, 0)
    
    return distances
    
    
if __name__ == '__main__':
    parser = arglib.ArgumentParser("""Takes a trajectory (the input data,
'INPUT') and a PDB, and calculates the RMSD between every frame of the trajectory
and PDB for the atoms specified in the atom indicies file. Note that trajectory
can be any trajectory-like format, including generators and random conformation 
files. Output: a flat file vector of RMSDs, in nm. Note that MSMBuilder's RMSD
calculator is highly optimized, so this calculation should be rapid. Output: 
RMSD.dat, a flat text file of the RMSDs.""")
    parser.add_argument('pdb', type=arglib.TrajectoryType)
    parser.add_argument('input', description='Path to a trajectory-like file')
    parser.add_argument('project')
    parser.add_argument('atom_indices', description='Indices of atoms to compare',
        type=arglib.LoadTxtType(dtype=int), default='AtomIndices.dat')
    parser.add_argument('output', description='Flat text file for the output',
        default='RMSD.dat')
    args = parser.parse_args()
    
    arglib.die_if_path_exists(args.output)
    
    distances = run(args.project, args.pdb, args.input, args.atom_indices)
    print 'Saving Output: %s' % args.output
    np.savetxt(args.output, distances)


