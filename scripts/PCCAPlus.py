#!/usr/bin/env python
"""PCCAPlus.py Use PCCA+ to create a macrostate MSM.

Notes:

You may have to trim the TER and CRYSTAL entries from the PDB file due to MSMBuilder's limited PDB reading.

"""

import argparse
import sys

import scipy.io
import numpy as np

import ArgLib
from Emsmbuilder import Serializer, MSMLib, NewPCCAPlus

print __doc__

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-T','--TProb',  help='Filename of input tProb file.')
parser.add_argument('-a','--Ass',    help='Filename of input Assignments file.')
parser.add_argument('-w','--OutDir', help='Directory to output results.')
parser.add_argument('-M','--nMacro', help='Number of Macrostates.')
parser.add_argument('-F','--FluxCutoff', help='Discard eigenvectors below this flux (default: None).',default=None)
parser.add_argument('-u','--UsePCCA', help='Use Normal PCCA (default: plus).',default="plus")

def run(macrostates, Assignments, TC,OutDir="./Data/",FluxCutoff=None,UsePCCA=False):

    MacroAssFilename=OutDir+"/MacroAssignments.h5"
    ArgLib.CheckPath(MacroAssFilename)

    MacroMapFilename=OutDir+"/MacroMapping.dat"
    ArgLib.CheckPath(MacroMapFilename)
    if UsePCCA=="plus":
        print "Running PCCA+..."
        MAP = NewPCCAPlus.pcca_plus(TC,macrostates,flux_cutoff=FluxCutoff,do_minimization=False)[3]
        #MAP = NewPCCAPlus.iterative_pcca_plus(TC,macrostates,Assignments,population_cutoff=FluxCutoff,do_minimization=False,LagTime=1)[3]
    else:
        print "Running PCCA..."
        MAP = MSMLib.PCCA(TC, macrostates, FluxCutoff=FluxCutoff)
        
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
    FluxCutoff=args["FluxCutoff"]
    UsePCCA=args["UsePCCA"]

    if FluxCutoff!=None:
        FluxCutoff=float(FluxCutoff)
    
    OutDir=args["OutDir"]
    nMacro=int(args["nMacro"])
    print sys.argv

    Assignments=Serializer.LoadData(AssFilename)

    TMatrix = scipy.io.mmread(TFilename)

    run(nMacro, Assignments, TMatrix, OutDir=OutDir,FluxCutoff=FluxCutoff,UsePCCA=UsePCCA)
