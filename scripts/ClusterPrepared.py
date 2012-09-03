#!/home/schwancr/epd-7.1-1-rh5-x86_64/bin/python -u
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
from schwancrtools import ArgLib_E as ArgLib, sshfs_tools
from msmbuilder import Project
from msmbuilder import Trajectory as Trajectory_msm
from schwancrtools.Trajectory_crs import Trajectory
from schwancrtools.clustering import HybridKMedoids, deterministic_subsample
from schwancrtools.metrics_PCA import RedDimPNorm
from msmbuilder import Serializer

def main():
    print """
    \nClusters data based on RMSD. Implements k-centers algorithm, followed by
    hybrid k-medoids to clean up data.

    Output: Generators in a Trajectory format (Gens.lh5).\n

    To use other distance metrics or clustering algorithms, modify this script
    or see the examples
    """

    arglist=["projectfn", "clusters", "stride", 
             "clustcutoff","outdir","atomindices"]
    new_arglist = [ ('localkmedoids','--lm','--local-k-med','Number of local k-medoids iterations to do. Default: 0',0),
                    ('globalkmedoids','--gm','--global-k-med','Number of global k-medoids iterations to do. Default: 0',0),
                    ]
    options, metric = ArgLib.parse(arglist,new_arglist=new_arglist,metric_parsers=True)
    print sys.argv

    # parse command line options    
    proj = Project.LoadFromHDF(options.projectfn)
    gens_path = os.path.join(options.outdir, "Gens.lh5")
    stride = int(options.stride)
    k = int(options.clusters)
    clustcutoff = float(options.clustcutoff)
    local_iterations = int(options.localkmedoids)
    global_iterations = int(options.globalkmedoids)
    try: aind = np.loadtxt( options.atomindices ).astype(int)
    except: aind = None
    ArgLib.CheckPath(gens_path)

    # load up all the trajectories
    #trajs = [proj.LoadTraj(i) for i in range(proj['NumTrajs'])]
    print "Loading Trajectories"
    trajs = []
    which_inds = []
    for i in range(proj['NumTrajs']):
        print i
        temp=Trajectory.LoadFromLHDF( proj.GetTrajFilename(i), Stride=stride, AtomIndices=aind )
        ptemp = metric.prepare_trajectory(temp)
        frame_inds = range( proj['TrajLengths'][i] )[::stride]
        which_inds.extend( zip( [i]*len(frame_inds), frame_inds ) )
        #temp['XYZList'] = temp['XYZList'][::stride]
        trajs.append(ptemp)
        del temp
 
        if ( not (i+1)%10 ):
            sshfs_tools.remount()
    
    which_inds = np.array( which_inds )

    sshfs_tools.remount()

    ptrajs = np.concatenate( trajs )
    #stride
    #subsampled = deterministic_subsample(trajs, stride)
    #subsampled=trajs   
    #print sum([ len(t) for t in trajs ]) 
    # cluster
    
    print "Clustering..."
    clusterer = HybridKMedoids(metric, ptrajs, k, clustcutoff,
                               local_iterations, global_iterations,already_prepared=True)

    gens_which = which_inds[ clusterer._generator_indices.astype(int) ]
    gens = proj.GetConformations( Which = gens_which )
    #gens = clusterer.get_generators_as_traj()

    print clusterer._generator_indices
    # save to disk

    np.savetxt(os.path.join(options.outdir,'which_frames.dat'),which_inds,'%d')
    np.savetxt(os.path.join(options.outdir,'which_gens.dat'),gens_which,'%d')
    np.savetxt(os.path.join(options.outdir,'gen_inds.dat'),clusterer._generator_indices,'%d')
    gens.SaveToLHDF(gens_path)
    print "Saved data to %s" % gens_path

    if hasattr( options, 'gen_eps'):
        gen_eps = clusterer.ptraj['epsilons'][ clusterer._generator_indices ]
        np.savetxt( options.gen_eps, gen_eps )
        print "Saved generator's epsilon values to %s" % options.gen_eps

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
    






