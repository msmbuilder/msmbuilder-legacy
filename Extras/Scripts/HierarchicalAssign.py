#!/usr/bin/env python
"""Assign data using a hierarchical clustering.
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

import HierarchicalClustering
from msmbuilder import Project, Serializer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p','--ProjectFn',    help='(input) Project Filename (ProjectInfo.h5)', default="ProjectInfo.h5")
    parser.add_argument('-c','--ClusteringFn', help='(input) Clustering filename (ZMatrix.h5)', default="ZMatrix.h5")
    parser.add_argument('-n','--NumStates',    help='Number of States')
    parser.add_argument('-a','--AssignmentsFn',help='(output) Assignments Filename (Assignments.h5)',default="Assignments.h5")
    args = vars(parser.parse_args())

    ProjectFilename=args["ProjectFn"]
    ClusteringFilename=args["ClusteringFn"]
    NumStates=args["NumStates"]
    AssignmentsFilename=args["AssignmentsFn"]

    P1=Project.Project.LoadFromHDF(ProjectFilename)    
    Z=Serializer.LoadData(ClusteringFilename)
    Assignments=HierarchicalClustering.AssignUsingScipy(P1,Z,NumStates)
    Serializer.SaveData(AssignmentsFilename,Assignments)
