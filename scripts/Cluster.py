#!/usr/bin/env python


import sys, os
import pickle
from pprint import pprint
from msmbuilder import arglib
from msmbuilder import metrics
from msmbuilder import clustering
from msmbuilder import Project
from msmbuilder import Serializer
from msmbuilder.utils import format_block
from msmbuilder.License import LicenseString
from msmbuilder.Citation import CiteString 
from msmbuilder.arglib import ensure_path_exists, die_if_path_exists
import argparse
import numpy as np

def add_argument(group, *args, **kwargs):
    if 'default' in kwargs:
        d = 'Default: {d}'.format(d=kwargs['default'])
        if 'help' in kwargs:
            kwargs['help'] += ' {d}'.format(d=d)
        else:
            kwargs['help'] = d
    group.add_argument(*args, **kwargs)

################################################################################

parser = arglib.ArgumentParser(description='''
    Cluster.py: Cluster MD trajectories into microstates
    
    Output: Assignments.h5, and other files depending on your choice of distance
    metric and/or clustering algorithm.
    
    Note that there are many distance metrics and clustering algorithms available
    Many of which have multiple options and parameters.
    
    MAKE LIBERAL USE OF THE -h OPTION. The help text changes significantly
    depending on which level in the options tree you are currently in''',get_metric=True)
parser.add_argument('project')
parser.add_argument( dest='stride', help='Subsample by striding',
    default=1, type=int)
parser.add_argument( dest='assignments', help='''Output assignments file
    (will be used if stride is 1 and you're not using hierarchical)''',
    default='Data/Assignments.h5')
parser.add_argument( dest='distances', help='''Output assignments distances file.
    Will be used if stride is 1 and you're not using hierarchical.
    (distance from each data point to its cluster center according to your selected
    distance metric). Note that for hierarchical clustering methods, this file will
    not be produced.''', default='Data/Assignments.h5.distances')
parser.add_argument( dest='generators', help='''Output trajectory file containing
    the structures of each of the cluster centers. Note that for hierarchical clustering
    methods, this file will not be produced.''', default='Data/Gens.lh5')

################################################################################

for metric_parser in parser.metric_parsers: # arglib stores the metric subparsers in that list

    subparser = metric_parser.add_subparsers( description='''Choose one of the following 
        clustering algorithms.''', dest='alg' )

    kcenters = subparser.add_parser('kcenters') 
#    kcenters.set_defaults(alg='kcenters', metric=subparser.metric)
    add_argument(kcenters, '-s', help='seed for initial cluster center.', default=0, type=int, dest='kcenters_seed')
    kcenters_cutoff = kcenters.add_argument_group('cutoff (use one)').add_mutually_exclusive_group(required=True)
    add_argument(kcenters_cutoff, '-k', help='number of clusters',
        type=int, dest='kcenters_num_clusters')
    add_argument(kcenters_cutoff, '-d', help='no greater cophenetic distance than this cutoff',
        type=float, dest='kcenters_distance_cutoff')
    
    
    hybrid = subparser.add_parser('hybrid')
#    hybrid.set_defaults(alg='hybrid', metric=subparser.metric)
    add_argument(hybrid, '-l', dest='hybrid_local_num_iters', default=10, type=int)
    add_argument(hybrid, '-g', dest='hybrid_global_iters', default=0, type=int)
    add_argument(hybrid, '-i', dest='hybrid_ignore_max_objective', type=bool, default=False)
    add_argument(hybrid, '-t', dest='hybrid_too_close_cutoff', default=0.0001, type=float)
    hybrid_cutoff = hybrid.add_argument_group('cutoff (use one)').add_mutually_exclusive_group(required=True)
    add_argument(hybrid_cutoff, '-k', help='number of clusters',
        type=int, dest='hybrid_num_clusters')
    add_argument(hybrid_cutoff, '-d', help='no greater cophenetic distance than this cutoff',
        type=float, dest='hybrid_distance_cutoff')
    

    clarans = subparser.add_parser('clarans')
#    clarans.set_defaults(alg='clarans', metric=subparser.metric)
    claransR = clarans.add_argument_group('required')
    add_argument(claransR, '-k', help='number of clusters',
        type=int, dest='clarans_num_clusters')
    add_argument(clarans, '-u', dest='clarans_num_local_minima', default=10, type=int,
        help='Number of local minima to find.')
    add_argument(clarans, '-m', dest='clarans_max_neighbors', default=20, type=int,
        help='Max number of neighbors to search before declaring a minima.')
    add_argument(clarans, '-l', dest='clarans_local_swap', default=True, type=bool,
        help='Perform loval swaps or global swaps.')

    sclarans = subparser.add_parser('sclarans')
#    sclarans.set_defaults(alg='sclarans', metric=subparser.metric)
    sclaransR = sclarans.add_argument_group('required')
    add_argument(sclaransR, '-k', help='number of clusters',
        type=int, dest='sclarans_num_clusters', required=True)
    add_argument(sclaransR, '-n', help='number of samples to draw',
        type=int, dest='sclarans_num_samples', required=True)
    add_argument(sclaransR, '-s', help='shrink multiple',
        type=int, dest='sclarans_shrink_multiple', required=True)    
    add_argument(sclarans, '-u', dest='sclarans_num_local_minima', default=10, type=int,
        help='Number of local minima to find.')
    add_argument(sclarans, '-m', dest='sclarans_max_neighbors', default=20, type=int,
        help='Max number of neighbors to search before declaring a minima.')
    add_argument(sclarans, '-l', dest='sclarans_local_swap', default=True, type=bool,
        help='Perform loval swaps or global swaps.')
    add_argument(sclarans, '-p', dest='sclarans_parallel', choices=['multiprocessing', 'dtm', 'none'],
        help='Perform in parallel.', default='none')

    hier = subparser.add_parser('hierarchical')
#    hier.set_defaults(alg='hierarchical', metric=subparser.metric)
    add_argument(hier, '-m', default='ward', help='method. default=ward',
        choices=['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'], dest='hierarchical_method')
    add_argument(hier, '-o', dest='hierarchical_save_zmatrix', help='Save Z-matrix to disk', default='Data/Zmatrix.h5')

def load_trajectories(projectfn, stride):
    project = Project.LoadFromHDF(projectfn)
    #return [traj[::stride] for traj in project.EnumTrajs()]
    #The following code has improved memory usage.
    longtraj = []
    for i in xrange(project['NumTrajs']):
        t = project.LoadTraj(i)
        t.subsample(stride)
        longtraj.append(t)
    return longtraj
    
def cluster(metric, trajs, args):
    if args.alg == 'kcenters':
        clusterer = clustering.KCenters(metric, trajs, k=args.kcenters_num_clusters,
            distance_cutoff=args.kcenters_distance_cutoff, seed=args.kcenters_seed)
    elif args.alg == 'hybrid':
        clusterer = clustering.HybridKMedoids(metric, trajs, k=args.hybrid_num_clusters,
            distance_cutoff=args.hybrid_distance_cutoff,
            local_num_iters=args.hybrid_local_num_iters,
            global_num_iters=args.hybrid_global_iters,
            too_close_cutoff=args.hybrid_too_close_cutoff,
            ignore_max_objective=args.hybrid_ignore_max_objective)
    elif args.alg == 'clarans':
        clusterer = clustering.Clarans(metric, trajs, k=args.clarans_num_clusters,
            num_local_minima=args.clarans_num_local_minima,
            max_neighbors=args.clarans_max_neighbors,
            local_swap=args.clarans_local_swap)
    elif args.alg == 'sclarans':
        clusterer = clustering.SubsampledClarans(metric, trajs, k=args.sclarans_num_clusters,
            num_samples=args.sclarans_num_samples,
            shrink_multiple=args.sclarans_shrink_multiple,
            num_local_minima=args.sclarans_num_local_minima,
            max_neighbors=args.sclarans_max_neighbors,
            local_swap=args.sclarans_local_swap,
            parallel=args.sclarans_parallel)
    elif args.alg == 'hierarchical':
        clusterer = clustering.Hierarchical(metric, trajs, method=args.hierarchical_method)
        print 'Saving zmatrix to %s' % args.hierarchical_save_zmatrix
        clusterer.save_to_disk(args.hierarchical_save_zmatrix)
    else:
        raise ValueError('!')
    
    return clusterer
    

def check_paths(args):
    if args.alg == 'hierarchical':
        die_if_path_exists(args.hierarchical_save_zmatrix)
    else:
        die_if_path_exists(args.generators)
        if args.stride == 1:
            die_if_path_exists(args.assignments)
            die_if_path_exists(args.distances)

    
def main(args, metric):
    check_paths(args)
    
    if args.alg == 'sclarans' and args.stride != 1:
        print >> sys.stderr, """\nYou don't want to use a stride with sclarans. The whole point of
sclarans is to use a shrink multiple to accomplish the same purpose, but in parallel with
stochastic subsampling. If you cant fit all your frames into  memory at the same time, maybe you
could stride a little at the begining, but its not recommended."""
        sys.exit(1)
    
    trajs = load_trajectories(args.project, args.stride)
    print 'Loaded %d trajs' % len(trajs)
    
    clusterer = cluster(metric, trajs, args)
    
    if not isinstance(clusterer, clustering.Hierarchical):
        generators = clusterer.get_generators_as_traj()
        print 'Saving %s' % args.generators
        generators.SaveToLHDF(args.generators)
        if args.stride == 1:
            assignments = clusterer.get_assignments()
            distances = clusterer.get_distances()
            
            print 'Saving %s' % args.assignments
            print 'Saving %s' % args.distances
            Serializer.SaveData(args.assignments, assignments)
            Serializer.SaveData(args.distances, distances)

if __name__ == '__main__':

    args, metric = parser.parse_args()
    pprint(args.__dict__)
    
    if hasattr(args, 'sclarans_parallel')  and args.sclarans_parallel == 'dtm':
        from deap import dtm
        dtm.start(main, args)
    else:
        main(args, metric)


    
    
    
        
