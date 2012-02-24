#!/usr/bin/python
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


import os, sys
import numpy as np
from Emsmbuilder.scripts import ArgLib
from Emsmbuilder.Project import Project
from Emsmbuilder.clustering import HybridKMedoids, deterministic_subsample
from Emsmbuilder.metrics import RMSD
from Emsmbuilder import Serializer

def main():
    print """
    \nClusters data based on RMSD. Implements k-centers algorithm, followed by
    hybrid k-medoids to clean up data.

    Output: Generators in a Trajectory format (Gens.lh5).\n

    To use other distance metrics or clustering algorithms, modify this script
    or see the examples
    """

    arglist=["projectfn", "clusters", "stride", "atomindices",
             "globalkmediods", "localkmediods", "rmsdcutoff","outdir"]
    options = ArgLib.parse(arglist)
    print sys.argv

    # parse command line options    
    proj = Project.LoadFromHDF(options.projectfn)
    atom_inds = np.loadtxt(options.atomindices, np.int)
    gens_path = os.path.join(options.outdir, "Gens.lh5")
    stride = int(options.stride)
    k = int(options.clusters)
    rmsd_cutoff = float(options.rmsdcutoff)
    local_iterations = int(options.localkmediods)
    global_iteratiobs = int(options.globalkmediods)

    ArgLib.CheckPath(gens_path)

    # load up all the trajectories
    trajs = [proj.LoadTraj(i) for i in range(proj['NumTrajs'])]

    #stride
    subsampled = deterministic_subsample(trajs, stride)
        
    # cluster
    rmsd_metric = RMSD(atom_inds)
    clusterer = HybridKMedoids(rmsd_metric, subsampled, k, rmsd_cutoff,
                               local_iterations, global_iteratiobs)
    gens = clusterer.get_generators_as_traj()

    # save to disk
    gens.SaveToLHDF(gens_path)

    # if stride is 1, save the assignments too
    if stride == 1:
        print "Stride is one, so we're saving the assignments and distances as well"
        asgn_path = os.path.join(options.outdir, "Assignments.h5")
        dist_path = os.path.join(options.outdir, "Assignments.h5.RMSD")
        Serializer.SaveData(asgn_path, clusterer.get_assignments())
        Serializer.SaveData(dist_path, clusterer.get_distances())
    
    print 'All Done!'
        
if __name__ == '__main__':
    main()
    






