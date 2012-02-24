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

method_list=["average","centroid","complete","median","single","ward","weighted"]

import argparse
import numpy as np
import sys
import os

from Emsmbuilder import Serializer, metrics
from Emsmbuilder.clustering import KCenters, Hierarchical, Clarans
from Emsmbuilder.Project import Project

allowable_metrics=["rmsd","dihedrals"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p','--ProjectInfoFn',  help='(input) ProjectInfo Filename (default: ProjectInfo.h5)', default="ProjectInfo.h5")
    parser.add_argument('-a','--AtomIndicesFn',  help='(input) Atom Indices array filename (default: AtomIndices.dat)', default="AtomIndices.dat")
    parser.add_argument('-c','--ClusteringFn',help='(output) Clustering filename (default: ZMatrix.h5)', default="ZMatrix.h5")
    parser.add_argument('-m','--Method',  help='Clustering Algorithm (default: ward)', default="ward")
    parser.add_argument('-d','--Metric',  help='Distance metric (default: rmsd)', default="rmsd")
    args = vars(parser.parse_args())

    atom_indices_filename = args["AtomIndicesFn"]
    project_filename = args["ProjectInfoFn"]
    clustering_filename = args["ClusteringFn"]
    method = args["Method"]
    metric_name = args["Metric"]

    if method not in method_list:
        raise Exception("Method must be one of %s"%method_list)

    if metric_name not in allowable_metrics:
        raise Exception("Metric must be one of %s"%allowable_metrics)
    
    if metric_name=="rmsd":
        metric = metrics.RMSD(atomindices=np.loadtxt(atom_indices_filename,'int'))
    elif metric_name=="dihedrals":
        metric = metrics.Dihedral(metric='euclidean')

    project = Project.LoadFromHDF(project_filename)

    trajectories = [project.LoadTraj(i) for i in range(project['NumTrajs'])]

    hierarchical = Hierarchical(metric, trajectories, method=method)

    hierarchical.save_to_disk(clustering_filename)
