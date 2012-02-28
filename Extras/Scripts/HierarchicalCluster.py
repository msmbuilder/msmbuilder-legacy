#!/usr/bin/env python
"""Calculates the Hierarchical clustering Z matrix.
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

MethodList=["average","centroid","complete","median","single","ward","weighted"]

import argparse
import numpy as np
import sys
import os

import fastcluster
from msmbuilder import Serializer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i','--RMSDFn',  help='(input) Distance metric array filename', default="PairwiseRMSD.h5")
    parser.add_argument('-c','--ClusteringFn',help='(output) Clustering filename', default="ZMatrix.h5")
    parser.add_argument('-m','--Method',  help='Clustering Algorithm', default="ward")
    args = vars(parser.parse_args())

    RMSDFilename=args["RMSDFn"]
    ClusteringFilename=args["ClusteringFn"]
    Method=args["Method"]
    if Method not in MethodList:
        print("Error: Method must be one of %s"%MethodList)
        sys.exit()

    if Method not in MethodList:
        raise Exception("Method must be one of %s"%MethodList)
    
    RMSD=Serializer.LoadData(RMSDFilename)
    f=getattr(fastcluster,Method)
    Z=f(RMSD)
    Serializer.SaveData(ClusteringFilename,Z)
