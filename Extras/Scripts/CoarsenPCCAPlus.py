#!/usr/bin/env python
"""Eliminate states with populations below some cutoff.  
"""
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

import argparse
import os

import IterPCCA
import HierarchicalClustering
from msmbuilder import Serializer, MSMLib, Project
import scipy.io

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m','--MicroAssignments',  help='(input) Directory of Microstate model')
    parser.add_argument('-w','--OutDir',  help='(output) Directory to store results.',)
    parser.add_argument('-E','--PopulationCutoff',  help='(input) Discard states with population below E. (default 0.02)', default=0.02)
    parser.add_argument('-n','--NumMacro',  help='(input) Number of MacroStates')
    parser.add_argument('-i','--MaxIter',  help='(input) Max Number of PCCA+ attempts. (default 10)',default=10)
    parser.add_argument('-p','--ProjectFn', help='(input) Project Filename. (default ProjectInfo.h5)',default="ProjectInfo.h5")
    parser.add_argument('-r','--PairwiseRMSDFn', help='(input) Pairwise RMSD Filename. (default PairwiseRMSD.h5)',default="PairwiseRMSD.h5")
    
    args = vars(parser.parse_args())

    OutDir=args["OutDir"]
    MaxIter=int(args["MaxIter"])
    MicroAssFilename=args["MicroAssignments"]
    Epsilon=float(args["PopulationCutoff"])
    NumMacro=int(args["NumMacro"])
    PairwiseRMSDFn=args["PairwiseRMSDFn"]
    ProjectFn=args["ProjectFn"]

    P1=Project.Project.LoadFromHDF(ProjectFn)
    CL,EC=HierarchicalClustering.ConstructConfListing(P1)
    PairwiseRMSD=Serializer.LoadData("./PairwiseRMSD.h5")
    Am=Serializer.LoadData(MicroAssFilename)

    IterPCCA.CoarsenPCCAPlus(Am,NumMacro,PairwiseRMSD,CL,EC)

    os.mkdir(OutDir)
    Serializer.SaveData(OutDir+"/MicroAssignments.h5",Am)
