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
import numpy as np
from msmbuilder.metrics import RMSD
from msmbuilder import Trajectory
from msmbuilder import Serializer
from msmbuilder import arglib


def run(project, pdb, atom_indices):    
    distances = -1 * np.ones((project['NumTrajs'], max(project['TrajLengths'])))
    rmsd = RMSD(atom_indices)
    ppdb = rmsd.prepare_trajectory(pdb)
    
    for i in xrange(project['NumTrajs']):
        ptraj = rmsd.prepare_trajectory(project.LoadTraj(i))
        d = rmsd.one_to_all(ppdb, ptraj, 0)
        distances[i, 0:len(d)] = d
    
    return distances
    
    
if __name__ == '__main__':
    parser = arglib.ArgumentParser(description="""
Calculate the RMSD between an input PDB and all conformations in your project.
Output as a HDF5 file (load using Serializer.LoadData())""")
    parser.add_argument('pdb', type=arglib.TrajectoryType)
    parser.add_argument('atom_indices', description='Indices of atoms to compare',
        type=arglib.LoadTxtType(dtype=int), default='AtomIndices.dat')
    parser.add_argument('output', description='''Output file name. Output is an
        .h5 file with RMSD entries corresponding to the Assignments.h5 file.''',
        default='Data/RMSD.h5')
    parser.add_argument('project')
    args = parser.parse_args()
    
    arglib.die_if_path_exists(args.output)
    
    distances = run(args.project, args.pdb, args.atom_indices)
    
    print 'Saving', args.output
    Serializer.SaveData(args.output, distances)