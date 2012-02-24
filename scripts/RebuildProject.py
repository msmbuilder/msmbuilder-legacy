#!/usr/bin/env python
"""RebuildProject.py: Search for local trajectories and create a ProjectInfo.h5 file.

"""
import argparse
import os

from msmbuilder import Project,CreateMergedTrajectoriesFromFAH

print __doc__

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-s','--PDBFN', help='Template PDB Filename', default="native.pdb")
parser.add_argument('-f','--filetype', help='Filetype of trajectories to use.', default=".lh5")
parser.add_argument('-p','--projectfn', help='Filename of Project to output.', default="ProjectInfo.h5")
args = vars(parser.parse_args())

ProjectFilename=args["projectfn"]
PDBFilename=args["PDBFN"]
FileType=args["filetype"]

if not os.path.exists(ProjectFilename):
        P1=Project.CreateProjectFromDir(Filename=ProjectFilename,ConfFilename=PDBFilename,TrajFileType=FileType)
