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
import scipy.io
from numpy import savetxt

from Emsmbuilder import Serializer
from Emsmbuilder import MSMLib

import ArgLib

def run(macrostates, Assignments, TC, Simplex=False,OutDir="./Data/"):

    MacroAssFilename=OutDir+"/MacroAssignments.h5"
    ArgLib.CheckPath(MacroAssFilename)

    MacroMapFilename=OutDir+"/MacroMapping.dat"
    ArgLib.CheckPath(MacroMapFilename)
    

    # Calculate transition prob. and counts matricies
    if Simplex:
        print "Running PCCA+..."
        MAP = MSMLib.PCCA_Simplex(TC, macrostates, doMinimization=False)
    else:
        print "Running PCCA..."
        MAP = MSMLib.PCCA(TC,macrostates)

    # MAP the new assignments and save, make sure don't mess up negaitve one's (ie where don't have data)
    MSMLib.ApplyMappingToAssignments(Assignments,MAP)

    savetxt(MacroMapFilename,MAP,"%d")
    Serializer.SaveData(MacroAssFilename,Assignments)
    
    print "Wrote: %s, %s"%(MacroAssFilename,MacroMapFilename)
    return


if __name__ == "__main__":
    print """
\nApplies the PCCA algorithm to lump your microstates into macrostates. You may
specify a transition matrix if you wish - this matrix is used to determine the
dynamics of the microstate model for lumping into kinetically relevant
macrostates. Also, you can specify to use the simplex verision (PCCA+), which is
more robust but more computationally intesive.

Output: MacroAssignments.h5, a new assignments HDF file, for the Macro MSM.\n"""

    arglist=["macrostates", "assignments", "tmat", "outdir"] # TJL disabled "simplex"
    options=ArgLib.parse(arglist, Custom=[("assignments", "Path to assignments file. Default: Data/Assignments.Fixed.h5", "Data/Assignments.Fixed.h5")])
    print sys.argv

    Assignments=Serializer.LoadData(options.assignments)
    macrostates = int(options.macrostates)
    TMatrix = scipy.io.mmread(str(options.tmat))

    # TJL 9/2/11:
    # PCCA+ (i.e. simplex version) appears to not be robust. We have disabled this feature
    #if options.simplex == 'simplex': simplex = True
    #else: simplex = False
    simplex = False

    run(macrostates, Assignments, TMatrix, Simplex=simplex, OutDir=options.outdir)
