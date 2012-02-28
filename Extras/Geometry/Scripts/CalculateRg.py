#!/usr/bin/env python
"""CalculateRg.py: Calculates radius of gyration for a project.
"""
import argparse
from msmbuilder import Conformation, Trajectory,Serializer
from numpy import *
from Geometry import Geometry

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

Rg=P1.EvaluateObservableAcrossProject(Geometry.GetRg)

Serializer.SaveData("./Rg.h5",Rg)
