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
from Emsmbuilder.Project import Project
from Emsmbuilder.Trajectory import Trajectory
from Emsmbuilder.metrics import RMSD
from Emsmbuilder import Serializer
from Emsmbuilder.utils import format_block
from Emsmbuilder.scripts import ArgLib


def main():
    print format_block("""
    Calculate the RMSD between an input PDB and all conformations in your
    project.  Output as a HDF5 file (load using Serializer.LoadData()):
    default Data/RMSD.h5""")
    
    arglist = ["PDBfn", "atomindices", "output", "projectfn"]
    options = ArgLib.parse(arglist, Custom=[("output", "Output file name. Output is an .h5 file with RMSD entries corresponding to the Assignments.h5 file.", "./RMSD.h5")])
    print sys.argv
    if os.path.exists(options.output):
        print "Error: File %s already exists! Exiting." % options.output
        sys.exit(1)
    
    project = Project.LoadFromHDF(options.projectfn)
    pdb = Trajectory.LoadTrajectoryFile(options.PDBfn)
    atom_indices = np.loadtxt(options.atomindices, np.int)
    
    distances = -1 * np.ones((project['NumTrajs'], max(project['TrajLengths'])))
    rmsd = RMSD(atom_indices)
    ppdb = rmsd.prepare_trajectory(pdb)
    
    for i in xrange(project['NumTrajs']):
        ptraj = rmsd.prepare_trajectory(project.LoadTraj(i))
        d = rmsd.one_to_all(ppdb, ptraj, 0)
        distances[i, 0:len(d)] = d
    
    Serializer.SaveData(options.output, distances)
    
    
if __name__ == '__main__':
    main()