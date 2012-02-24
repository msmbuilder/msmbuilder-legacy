#!/usr/bin/env python
"""ConvertRepexToMSMBuilder.py: Convert a NetCDF replica exchange database into an MSMBuilder Project.

Notes:

You may have to trim the TER and CRYSTAL entries from the PDB file due to MSMBuilder's limited PDB reading.

"""

import argparse
import sys
import os

import scipy.io
import numpy as np

import ArgLib
from msmbuilder import Trajectory, Project, CreateMergedTrajectoriesFromFAH,Conformation, Serializer, MSMLib

print __doc__

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-T','--TProb',  help='Filename of input tProb file.')
parser.add_argument('-a','--Ass',    help='Filename of input Assignments file.')
parser.add_argument('-w','--OutDir', help='Directory to output results.')
parser.add_argument('-M','--nMacro', help='Number of Macrostates.')

def run(macrostates, Assignments, TC,OutDir="./Data/"):

    MacroAssFilename=OutDir+"/MacroAssignments.h5"
    ArgLib.CheckPath(MacroAssFilename)

    MacroMapFilename=OutDir+"/MacroMapping.dat"
    ArgLib.CheckPath(MacroMapFilename)
    
    print "Running PCCA+..."
    MAP = MSMLib.PCCA_Simplex(TC, macrostates, doMinimization=True)
    
    # MAP the new assignments and save, make sure don't mess up negaitve one's (ie where don't have data)
    MSMLib.ApplyMappingToAssignments(Assignments,MAP)

    np.savetxt(MacroMapFilename,MAP,"%d")
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

    args = vars(parser.parse_args())
    TFilename=args["TProb"]
    AssFilename=args["Ass"]
    
    OutDir=args["OutDir"]
    nMacro=int(args["nMacro"])
    print sys.argv

    Assignments=Serializer.LoadData(AssFilename)

    TMatrix = scipy.io.mmread(TFilename)

    run(nMacro, Assignments, TMatrix, OutDir=OutDir)
