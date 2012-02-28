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

import sys
import os
import numpy

from msmbuilder import Conformation
from msmbuilder import Trajectory

import ArgLib

def run(conformation, trajectory, atomindices, output):

    # Check output isn't taken
    if os.path.exists(output):
        print "Error: File %s already exists! Exiting." % output
        sys.exit(1)

    # Do RMSD calculation
    rmsd=trajectory.CalcRMSD(conformation,atomindices,atomindices)
    numpy.savetxt(output, rmsd)
    print "Wrote:", output

    return


if __name__ == "__main__":
    print """
\nTakes a trajectory (the input data, 'INPUT') and a PDB, and calculates the
RMSD between every frame of the trajectory and PDB for the atoms specified in
the atom indicies file. Note that trajectory can be any trajectory-like format,
including generators and random conformation files. Output: a flat file vector
of RMSDs, in nm. Note that MSMBuilder's RMSD calculator is highly optimized, so
this calculation should be rapid.

Output: RMSD.dat, a flat text file of the RMSDs.\n"""

    arglist=["PDBfn", "input", "atomindices", "output"]
    options=ArgLib.parse(arglist, Custom=[("input", "The path to a trajectory type (e.g. XRandomConfs.lh5) to calculate the RMSD of. Calculates the RMSD between all snapshots of the trajectory and the PDB file, and output a flat text file of floats in the same order as the snapshots in the trajectory. Units are nm!", None)])
    if options.output == 'NoOutputSet': output='RMSD.dat'
    else: output=options.output
    print sys.argv

    C1=Conformation.Conformation.LoadFromPDB(options.PDBfn)
    Traj=Trajectory.Trajectory.LoadTrajectoryFile(options.input,Conf=C1)
    AInds=numpy.loadtxt(options.atomindices, int)

    run(C1, Traj, AInds, output)
