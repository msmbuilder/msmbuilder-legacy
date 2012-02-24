import os, sys
import numpy as np

from Emsmbuilder import Serializer
from Emsmbuilder.clustering import KCenters, Hierarchical
from Emsmbuilder import metrics
from Emsmbuilder.Project import Project

# load up Project from ~/msmbuilder/tutorial/ProjectInfo.h5
# (location may be different on your machine)
path_to_project = os.path.join(os.environ['HOME'], 'msmbuilder/tutorial/ProjectInfo.h5')

# Instantiate a metric. Metrics are objects that extend
# Emsmbuilder.metrics.AbstractDistanceMetric and provide a
# prepare_trajectory() method to do the preprocessing on the trajectory
# (such as extracting the dihedral angles for the dihedral metric)
# and then various methods for computing the distance without or
# between prepared trajectories (one_to_all(), one_to_many(), all_pairwise(),
# etc)
metric = metrics.Dihedral(metric='euclidean', angles='phi')

# Here are some more example metrics.
# ====== # ====== # ====== # ====== # ====== # ====== 

# metric = metrics.ContinuousContact(metric='cityblock)
# metric = RMSD(atom_indices=[2,3,4,10])
# metric = metrics.AtomPairs(metric='minkowski', p=3, atom_pairs=[[1,2], [3,7]])
# metric = metrics.BooleanContact(metric='matching, contacts='all', scheme='closest-heavy')
# metric = metrics.Hybrid(base_metrics=[m1, m2, m3], weights=[0.6, 0.2, 0,2])
# metric = metrics.Rg(metric='sqeuclidean')

# ====== ## ====== # ====== # ====== # ====== # ====== 
# See the docs for more detail on how each work
# (the docstrings that you get from the ? operator in ipython should be pretty complete,
# i.e. In[1]: from Emsmbuilder import metrics
#
#      In[2]: metrics.BooleanContact?

# String Form:<class 'Emsmbuilder.metrics.BooleanContact'>
# Namespace:  Interactive
# File:       /Library/Frameworks/EPD64.framework/Versions/7.1/lib/python2.7/site-packages/Emsmbuilder/metrics.py
# Docstring:
# This is a concerete implementation of AbstractDistanceMetric by way of
# Vectorized, for calculating distances between frames based on their
# contact maps.
#
# Here each frame is represented as a vector of booleans representing whether
# the distance between pairs of residues is less than a cutoff...

#  [This goes on for another 30 lines]


project = Project.LoadFromHDF(path_to_project)

# load up all the trajectories in the project
trajectories = [project.LoadTraj(i) for i in range(project['NumTrajs'])]

# Note that if you want to subsample, you can easily do it yourself
# both of the methods below will give you a new trajectory that you can
# pass to any clustering algorithm with 10x less data

# from clustering import deterministic_subsample, stochastic_subsample
# subsampled1 = deterministic_subsample(trajectories, 10)
# subsampled2 = stochastic_subsample(trajectories, 10)

# But you should be careful to note that the distances and assignments
# you get directly out of the clusterer will only be for the data that
# was passed to it, not the original data. So you will have to assign into
# those generators with the assigning module.


# run kcenters clustering -- 100 states
kcenters = KCenters(metric, trajectories, k=100)
Serializer.SaveData('distances.h5', kcenters.get_distances())
Serializer.SaveData('assignments.h5', kcenters.get_assignments())
kcenters.get_generators_as_traj().SaveToLHDF("MyGens.lh5")
print "KCenters Clustering Finished"
print "distances.h5 and assignments.h5, MyGens.lh5 saved to disk"

# run hierarchical clustering -- 100 states
# this requires computing the all to all distance matrix, is obviously
# N^2 in the amount of data in your project, and can be more depending
# on the aglomeration scheme (i.e. metric='ward' or 'complete' or whathaveyou)
hierarchical = Hierarchical(metric, trajectories, method='ward')
# note the slightly different API for setting the number of states in Hierarchical
# clustering -- you supply k (or cutoff_distance) when getting the assignments
# out, not when creating the clusterer
assignments = hierarchical.get_assignments(k=100)
Serializer.SaveData('ward_assignments_100.h5', assignments)
assignments2 = hierarchical.get_assignments(k=1000)
Serializer.SaveData('ward_assignments_1000.h5', assignments2)
print "Ward clustering finished"
print "ward_assignments_100.h5 and ward_assignment_1000.h5 saved to disk"
