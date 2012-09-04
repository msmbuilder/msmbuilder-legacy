#!/usr/bin/env python
import sys, os
import numpy as np

#from msmbuilder import arglib
import argparse
from msmbuilder import Trajectory
from msmbuilder.scripts.Cluster import add_argument, construct_metric
from msmbuilder import arglib
from msmbuilder.assigning import assign_with_checkpoint
from msmbuilder import metrics
from msmbuilder import Project

def main():
    parser = arglib.ArgumentParser(description="""
Assign data that were not originally used in the clustering (because of
striding) to the microstates. This is applicable to all medoid-based clustering
algorithms, which includes all those implemented by Cluster.py except the
hierarchical methods. (For assigning to a hierarchical clustering, use 
AssignHierarchical.py)

Outputs:
-Assignments.h5
-Assignments.h5.distances

Assignments.h5 contains the assignment of each frame of each trajectory to a 
microstate in a rectangular array of ints. Assignments.h5.distances is an 
array of real numbers of the same dimension containing the distance (according 
to whichever metric you choose) from each frame to to the medoid of the 
microstate it is assigned to.""", get_metric=True)#, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument( 'project')
    parser.add_argument( dest='generators', help='''Output trajectory file containing
        the structures of each of the cluster centers. Note that for hierarchical clustering
        methods, this file will not be produced.''', default='Data/Gens.lh5')
    parser.add_argument( 'output_dir' )
    
    args, metric = parser.parse_args()
    
    assignments_path = os.path.join(args.output_dir, "Assignments.h5")
    distances_path = os.path.join(args.output_dir, "Assignments.h5.distances")
    lock_path = os.path.join(args.output_dir, "Assignments.lock")
    project = Project.LoadFromHDF(args.project)
    gens = Trajectory.LoadTrajectoryFile(args.generators)
    
    # this runs assignment and prints them to disk
    all_asgn, all_dist = assign_with_checkpoint(metric, project, gens, assignments_path, distances_path)

    print 'All Done!'

if __name__ == '__main__':
    main()
