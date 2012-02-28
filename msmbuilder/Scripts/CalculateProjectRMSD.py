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
from numpy import loadtxt

from msmbuilder import Project
from msmbuilder import Conformation
from msmbuilder import Serializer
import ArgLib

def run(C1,P1,AInd,OutFilename):
    # Check output isn't taken
    if os.path.exists(OutFilename):
        print "Error: File %s already exists! Exiting." % output
        sys.exit(1)

    RMSD=P1.CalcRMSDAcrossProject(C1,AInd,AInd)
    Serializer.SaveData(OutFilename,RMSD)

    print "Wrote:", OutFilename

if __name__ == "__main__":
    print """
Calculate the RMSD between an input PDB and all conformations in your
project.  Output as a HDF5 file (load using Serializer.LoadData()):
default Data/RMSD.h5"""

    arglist=["PDBfn", "atomindices", "output", "projectfn"]
    options=ArgLib.parse(arglist, Custom=[("output", "Output file name. Output is an .h5 file with RMSD entries corresponding to the Assignments.h5 file.", "NoOutputSet")])
    if options.output == 'NoOutputSet': output='./RMSD.h5'
    else: output=options.output
    print sys.argv

    P1=Project.Project.LoadFromHDF(options.projectfn)
    C1=Conformation.Conformation.LoadFromPDB(options.PDBfn)
    AInds=loadtxt(options.atomindices, int)

    run(C1, P1, AInds, output)
