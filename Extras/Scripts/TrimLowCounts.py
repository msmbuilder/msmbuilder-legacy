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

    MacroCountsFilename=MacroDir+"/tCounts.mtx"
    PopulationsFilename=MacroDir+"/Populations.dat"
    MacroAssFilename=MacroDir+"/Assignments.Fixed.h5"
    MapFilename=MacroDir+"/MacroMapping.dat"

    MicroCountsFilename=MicroDir+"/tCounts.mtx"
    MicroAssFilename=MicroDir+"/Assignments.Fixed.h5"

    m=np.loadtxt(MapFilename,"int")
    pi=np.loadtxt(PopulationsFilename)

    Ind=np.where(pi<Epsilon)[0]
    print("Trimming discards %f of the population."%pi[Ind].sum())

    MicroAss=Serializer.LoadData(MicroAssFilename)
    CMicro=scipy.io.mmread(MicroCountsFilename).toarray()

    MacroAss=Serializer.LoadData(MacroAssFilename)
    CMacro=scipy.io.mmread(MacroCountsFilename).toarray()

    for i in Ind:
        CMacro[:,i]=0.
        CMacro[i,:]=0.

    for i in Ind:
	for j in np.where(m==i)[0]:
            CMicro[:,j]=0.
            CMicro[j,:]=0.

    CMicro=scipy.sparse.csr_matrix(CMicro)
    CMacro=scipy.sparse.csr_matrix(CMacro)

    CMicro2,Unused=MSMLib.ErgodicTrim(CMicro,MicroAss)
    CMacro2,Unused=MSMLib.ErgodicTrim(CMacro,MacroAss)

    os.mkdir(OutDir)
    Serializer.SaveData(OutDir+"/MacroAssignments.h5",MacroAss)
    Serializer.SaveData(OutDir+"/MicroAssignments.h5",MicroAss)
