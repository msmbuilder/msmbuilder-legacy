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
from msmbuilder.metrics import RMSD
from msmbuilder.metric_LPRMSD import LPRMSD, LPTraj, ReadPermFile
from msmbuilder import arglib

def run(project, pdb, traj_fn, atom_indices, alt_indices, permute_indices):

    #project = Project.LoadFromHDF(options.projectfn)
    traj = Trajectory.LoadTrajectoryFile(traj_fn,Conf=project.Conf)

    # you could replace this with your own metric if you like
    metric = LPRMSD(atom_indices, permute_indices, alt_indices)
    ppdb = metric.prepare_trajectory(pdb)
    ptraj = metric.prepare_trajectory(traj)

    print ppdb['XYZList'].shape
    print ptraj['XYZList'].shape

    distances, xout = metric.one_to_all_aligned(ppdb, ptraj, 0)
    print distances
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

    parser.add_argument('lprmsd_alt_indices', description='''Optional
    alternate atom indices for RMSD. If you want to align the
    trajectories using one set of atom indices but then compute the
    distance using a different set of indices, use this option. If
    supplied, the regular atom_indices will be used for the alignment
    and these indices for the distance calculation''', 
                        type=arglib.LoadTxtType(dtype=int), default='AltIndices.dat')

    parser.add_argument('lprmsd_permute_atoms', default='None', 
    description='''Atom labels to be permuted.  Sets of
    indistinguishable atoms that can be permuted to minimize the
    RMSD. On disk this should be stored as a list of newline separated
    indices with a "--" separating the sets of indices if there are
    more than one set of indistinguishable atoms''')


    parser.add_argument('output', description='Flat text file for the output',
        default='RMSD.dat')
    args = parser.parse_args()
    
    if args.lprmsd_permute_atoms == 'None':
        permute_indices = None
    else:
        permute_indices = ReadPermFile(args.lprmsd_permute_atoms)

    arglib.die_if_path_exists(args.output)
    
    distances = run(args.project, args.pdb, args.input, args.atom_indices, args.lprmsd_alt_indices, permute_indices)
    print 'Saving Output: %s' % args.output
    np.savetxt(args.output, distances)


