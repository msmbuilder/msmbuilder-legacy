#!/usr/bin/env python
"""Calculate a pairwise distance from backbone torsion angles.
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
import sys
from numpy import loadtxt,arange

from msmbuilder import Project, Conformation, Serializer
import DihedralTools

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p','--ProjectFn',    help='(input) Project Filename (ProjectInfo.h5)', default="ProjectInfo.h5")
    parser.add_argument('-r','--TorsionFn',help='(input) Filename of torsions', default="Torsions.h5")
    parser.add_argument('-o','--PairwiseTorsionFn',  help='(output) Distance metric array filename', default="PairwiseTorsion.h5")
    args = vars(parser.parse_args())

    ProjectFilename=args["ProjectFn"]
    RamaFilename=args["TorsionFn"]
    PairwiseTorsionFilename=args["PairwiseTorsionFn"]

    P1=Project.Project.LoadFromHDF(ProjectFilename)
    
    Rama=Serializer.LoadData(RamaFilename)
    PairwiseTorsions=DihedralTools.CalculatePairwiseTorsionMatrix(P1,Rama).astype('float32')
    Serializer.SaveData(PairwiseTorsionFilename,PairwiseTorsions)
