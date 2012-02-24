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
from Emsmbuilder.scripts import ArgLib
from Emsmbuilder.metrics import RMSD
from Emsmbuilder.utils import format_block

def main():
    print format_block("""Takes a trajectory (the input data, 'INPUT') and a PDB, and calculates the
    RMSD between every frame of the trajectory and PDB for the atoms specified in
    the atom indicies file. Note that trajectory can be any trajectory-like format,
    including generators and random conformation files. Output: a flat file vector
    of RMSDs, in nm. Note that MSMBuilder's RMSD calculator is highly optimized, so
    this calculation should be rapid.
    Output: RMSD.dat, a flat text file of the RMSDs.""")
    
    arglist = ["PDBfn", "input", "atomindices", "output", "projectfn"]
    options = ArgLib.parse(arglist, Custom=[("input", "The path to a trajectory type (e.g. XRandomConfs.lh5) to calculate the RMSD of. Calculates the RMSD between all snapshots of the trajectory and the PDB file, and output a flat text file of floats in the same order as the snapshots in the trajectory. Units are nm!", None),
                                            ("output", "Flat text file for the output", "RMSD.dat")])
    print sys.argv
    if os.path.exists(options.output):
        print "Error: File %s already exists! Exiting." % options.output
        sys.exit(1)
    
    project = Project.LoadFromHDF(options.projectfn)
    pdb = Trajectory.LoadFromPDB(options.PDBfn)
    traj = Trajectory.LoadTrajectoryFile(options.input,Conf=project.Conf)
    atom_indices = np.loadtxt(options.atomindices, np.int)
    
    # you could replace this with your own metric if you like
    metric = RMSD(atom_indices)

    ppdb = metric.prepare_trajectory(pdb)
    ptraj = metric.prepare_trajectory(traj)
    distances = metric.one_to_all(ppdb, ptraj, 0)
    
    np.savetxt(options.output, distances)
    
    
if __name__ == '__main__':
    main()
