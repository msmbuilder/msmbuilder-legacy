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

import numpy as np
import scipy.io, scipy.sparse
from msmbuilder import Serializer,MSMLib

import IterPCCA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-M','--MacroDir',  help='(input) Directory of Macrostate model')
    parser.add_argument('-m','--MicroDir',  help='(input) Directory of Microstate model')
    parser.add_argument('-w','--OutDir',  help='(output) Directory to store results.',)
    parser.add_argument('-E','--PopulationCutoff',  help='(input) Discard states with population below E. (default 0.02)', default=0.02)
    
    args = vars(parser.parse_args())

    OutDir=args["OutDir"]
    MicroDir=args["MicroDir"]
    MacroDir=args["MacroDir"]
    Epsilon=args["PopulationCutoff"]

    Cm=scipy.io.mmread(MicroDir+"/tCounts.mtx").tocsr()
    CM=scipy.io.mmread(MacroDir+"/tCounts.mtx").tocsr()

    AM=Serializer.LoadData(MacroDir+"./Assignments.Fixed.h5")
    Am=Serializer.LoadData(MicroDir+"./Assignments.Fixed.h5")

    Map=np.loadtxt(MacroDir+"./MacroMapping.dat","int")

    IterPCCA.TrimLowCounts(Cm,CM,Am,AM,Map)

    os.mkdir(OutDir)
    Serializer.SaveData(OutDir+"/Assignments.h5",Am)
