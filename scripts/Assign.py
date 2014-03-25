#!/usr/bin/env python
##############################################################################
# imports
##############################################################################

import sys
import os
import numpy as np

#from msmbuilder import arglib
import argparse
from msmbuilder import arglib
from msmbuilder.assigning import assign_with_checkpoint
from msmbuilder import metrics
from msmbuilder import Project
import logging
import mdtraj as md

##############################################################################
# Globals
##############################################################################

logger = logging.getLogger('msmbuilder.scripts.Assign')
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
microstate it is assigned to.""", get_metric=True)  # , formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('project')
parser.add_argument( dest='generators', help='''Output trajectory file containing
    the structures of each of the cluster centers. Note that for hierarchical clustering
    methods, this file will not be produced.''', default='Data/Gens.h5')
parser.add_argument('output_dir')


def main(args, metric):
    assignments_path = os.path.join(args.output_dir, "Assignments.h5")
    distances_path = os.path.join(args.output_dir, "Assignments.h5.distances")

    # arglib.die_if_path_exists(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    project = Project.load_from(args.project)
    gens = md.load(args.generators)

    if isinstance(metric, metrics.RMSD):
        # this is really bad design, and we're going to fix it soon in
        # MSMBuilder3, but here's the deal. When Cluster.py loads up the
        # trajectories (Cluster.py:load_trajectories()), it only loads the
        # required indices for RMSD. This means that when it saves the Gens
        # file, that file contains only a subset of the atoms. So when
        # we run *this* script, we need to perform a restricted load of the
        # the trajectories on disk, but we need to NOT perform a restricted
        # load of the gens.h5 file. (By restricted load, I mean loading
        # only a subset of the data in the file)
        if gens.n_atoms != len(metric.atomindices):
            msg = ('Using RMSD clustering/assignment, this script expects '
                   'that the Cluster.py script saves a generators file that '
                   'only contains the indices of the atoms of interest, and '
                   'not any of the superfluous degrees of freedom that were '
                   'not used for clustering. But you supplied %d cluster '
                   'centers each containg %d atoms. Your atom indices file '
                   'on the other hand contains %d atoms') \
                % (gens.xyz.shape[0], gens.xyz.shape[1],
                   len(metric.atomindices))
            raise ValueError(msg)

        # now that we're telling the assign function only to load up a
        # subset of the atoms, an the generator is already only a subset,
        # the actual RMSD object needs to, from ITS perspective, operate on
        # every degree of freedom. So it shouldn't be aware of any special
        # atom_indices
        atom_indices = metric.atomindices
        metric.atomindices = None
        # this runs assignment and prints them to disk
        assign_with_checkpoint(metric, project, gens, assignments_path,
                               distances_path, atom_indices_to_load=atom_indices)
    else:
        assign_with_checkpoint(metric, project, gens, assignments_path,
                               distances_path)

    logger.info('All Done!')

def entry_point():
    args, metric = parser.parse_args()
    main(args, metric)

if __name__ == '__main__':
    entry_point()
