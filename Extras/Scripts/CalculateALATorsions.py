#!/usr/bin/env python
"""CalculateALATorsions.py: a HACK that allows calculation of torsion angles in capped dipeptide.

"""
import argparse
from msmbuilder import Conformation, Trajectory,Serializer
import DihedralTools
from numpy import *

from msmbuilder import Project,CreateMergedTrajectoriesFromFAH

print __doc__

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-s','--PDBFN', help='Template PDB Filename', default="native.pdb")
parser.add_argument('-p','--projectfn', help='Filename of Project to output.', default="ProjectInfo.h5")
args = vars(parser.parse_args())

ProjectFilename=args["projectfn"]
PDBFilename=args["PDBFN"]


C1=Conformation.Conformation.LoadFromPDB(PDBFilename)
P1=Project.Project.LoadFromHDF(ProjectFilename)

AllPhi=[]
AllPsi=[]
for i in xrange(P1.GetNumTrajectories()):
    C1=Conformation.Conformation.LoadFromPDB(PDBFilename)
    R1=P1.LoadTraj(i)
    Phi=DihedralTools.GetTorsions(R1,C1,"Phi")[0][0]
    Ind=where(R1["ResidueNames"]!="ACE")[0]
    R1.RestrictAtomIndices(Ind)
    C1.RestrictAtomIndices(Ind)
    Psi=DihedralTools.GetTorsions(R1,C1,"Psi")[0][0]
    AllPhi.append(Phi)
    AllPsi.append(Psi)

AllPhi=array(AllPhi)
AllPsi=array(AllPsi)

Serializer.SaveData("./Phi.h5",AllPhi)
Serializer.SaveData("./Psi.h5",AllPsi)
