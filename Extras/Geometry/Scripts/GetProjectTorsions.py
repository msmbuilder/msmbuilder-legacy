#!/usr/bin/env python
"""Calculate backbone torsions for every frame in your project.
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
from numpy import loadtxt,arange,zeros,concatenate

from msmbuilder import Project, Conformation, Serializer
import DihedralTools

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p','--ProjectFn',    help='(input) Project Filename (ProjectInfo.h5)', default="ProjectInfo.h5")
    parser.add_argument('-o','--TorsionFn',  help='(output) Tosrions array', default="Torsions.h5")
    args = vars(parser.parse_args())
    
    ProjectFilename=args["ProjectFn"]
    OutFilename=args["TorsionFn"]

    P1=Project.Project.LoadFromHDF(ProjectFilename)
    C1=Conformation.Conformation.LoadFromPDB(P1["ConfFilename"])
    for i in range(P1.GetNumTrajectories()):
	print(i)
	R1=P1.LoadTraj(i)
	ans1=DihedralTools.GetTorsions(R1,C1,"Phi")
	ans2=DihedralTools.GetTorsions(R1,C1,"Psi")
	ans=concatenate((ans1,ans2))[:,0]
	if i==0:
		Result=zeros((P1.GetNumTrajectories(),P1["TrajLengths"].max(),ans.shape[0]),dtype='float32')
	Result[i,0:ans.shape[1]]=ans.transpose()

    Serializer.SaveData(OutFilename,Result)

