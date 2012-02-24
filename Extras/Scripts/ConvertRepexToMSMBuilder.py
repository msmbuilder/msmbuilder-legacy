#!/usr/bin/env python
"""ConvertRepexToMSMBuilder.py: Convert a NetCDF replica exchange database into an MSMBuilder Project.

Notes:

You may have to trim the TER and CRYSTAL entries from the PDB file due to MSMBuilder's limited PDB reading.

"""

import argparse
import os
import numpy as np

from msmbuilder import Trajectory, Project, CreateMergedTrajectoriesFromFAH,Conformation
import netCDF4 as netcdf
import sys
import os

print __doc__

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-s','--PDBFN', help='Template PDB Filename', default="native.pdb")
parser.add_argument('-f','--filetype', help='Filetype of trajectories to use.', default=".lh5")
parser.add_argument('-p','--projectfn', help='Filename of Project to output.', default="ProjectInfo.h5")
parser.add_argument('-c','--cdffn', help='Name of repex NetCDF file.', default="repex.nc")
parser.add_argument('-o','--outpdb', help='Name of output (trimmed) PDB file.', default="Trimmed.pdb")
args = vars(parser.parse_args())

ProjectFilename=args["projectfn"]
PDBFilename=args["PDBFN"]
FileType=args["filetype"]
CDFFilename=args["cdffn"]
OutputPDBFilename=args["outpdb"]

try:
    os.mkdir("./Trajectories/")
except:
    print("Error: ./Trajectories/ already detected.")
    sys.exit()

d=netcdf.Dataset(CDFFilename)
R1=Trajectory.Trajectory.LoadFromPDB(PDBFilename)
C1=Conformation.Conformation.LoadFromPDB(PDBFilename)

#Find index to exclude solvent residues
r=R1["ResidueNames"]
NameList=np.array(["HOH","SOL","WAT"])
Ind=np.where(~np.in1d(r,NameList))[0]
R1.RestrictAtomIndices(Ind)
C1.RestrictAtomIndices(Ind)

#Save a PDB with solvent removed
C1.SaveToPDB(OutputPDBFilename)


#x=d.variables["positions"][:]
x=d.variables["positions"][:,:,Ind,:]
XYZ=x
NumAtoms=R1.GetNumberOfAtoms()
NumDimensions=x.shape[-1]

#XYZ=x[:,:,Ind,:]

XYZ2=[]
for replica in xrange(XYZ.shape[1]):
    R1["XYZList"]=XYZ[:,replica]
    R1.SaveToLHDF("./Trajectories/trj%d.lh5"%replica)

P1=Project.CreateProjectFromDir(Filename=ProjectFilename,ConfFilename=OutputPDBFilename,TrajFileType=FileType)
