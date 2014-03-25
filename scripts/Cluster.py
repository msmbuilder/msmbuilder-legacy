#!/usr/bin/env python
from __future__ import print_function, absolute_import, division
from mdtraj.utils.six.moves import xrange
import sys, os
import warnings
import numpy as np
import mdtraj as md
from mdtraj import io
from msmbuilder import arglib
from msmbuilder import clustering
from msmbuilder import Project
from msmbuilder import metrics
from msmbuilder.arglib import die_if_path_exists
from msmbuilder.utils import highlight
import logging
logger = logging.getLogger('msmbuilder.scripts.Cluster')

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
    Cluster MD trajectories into microstates using a geometric criterion.
    
    Output: Assignments.h5, and other files depending on your choice of distance
    metric and/or clustering algorithm.
    
    Note that there are many distance metrics and clustering algorithms available
    Many of which have multiple options and parameters. 

    ''' + highlight('''MAKE LIBERAL USE OF THE -h OPTION. The help text changes significantly 
    depending on which level in the options tree you are currently in''', color='green', bold=True),get_metric=True)
parser.add_argument('project')
parser.add_argument(dest='stride', help='Subsample by striding',
    default=1, type=int)
parser.add_argument('output_dir', help='''Output directory to save clustering data.
    This will include some of the following depending on the clustering algorithm:
    (1) Assignments.h5 (If clustering is not hierarchical and stride=1):
        Contains the state assignments
    (2) Assignments.h5.distances (If clustering is not hierarchical and stride=1):
        Contains the distance to the generator according to the distance
        metric that was employed
    (3) Gens.h5 (If clustering is not hierarchical): 
        Trajectory object representing the generators for each state
    (4) ZMatrix.h5 (If clustering is hierarchical):
        This is the ZMatrix corresponding to the result of hierarchical clustering.
        use it with AssignHierarchical.py to build your assignments file.''')

################################################################################

for metric_parser in parser.metric_parser_list: # arglib stores the metric subparsers in that list

    subparser = metric_parser.add_subparsers( description='''Choose one of the following 
        clustering algorithms.''', dest='alg' )

    kcenters = subparser.add_parser('kcenters') 
    add_argument(kcenters, '-s', help='seed for initial cluster center.', default=0, type=int, dest='kcenters_seed')
    kcenters_cutoff = kcenters.add_argument_group('cutoff (use one)').add_mutually_exclusive_group(required=True)
    add_argument(kcenters_cutoff, '-k', help='number of clusters',
        type=int, dest='kcenters_num_clusters')
    add_argument(kcenters_cutoff, '-d', help='no greater cophenetic distance than this cutoff',
        type=float, dest='kcenters_distance_cutoff')
    
    
    hybrid = subparser.add_parser('hybrid')
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
    add_argument(hier, '-m', default='ward', help='method. default=ward',
        choices=['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'], dest='hierarchical_method')


def load_prep_trajectories(project, stride, atom_indices, metric):
    """load the trajectories but prepare them during the load.
    This is helpful for metrics that use dimensionality reduction
    so you can use more frames without a MemoryError
    """
    list_of_ptrajs = []
    which = []
    for i in xrange(project.n_trajs):

        which_frames = np.arange(0, project.traj_lengths[i], stride)

        which.extend(zip([i] * len(which_frames), which_frames))

        ptraj = []
        for chunk in md.iterload(project.traj_filename(i), stride=stride, atom_indices=atom_indices):
            ptrj_chunk = metric.prepare_trajectory(chunk)
            ptraj.append(ptrj_chunk)

        ptraj = np.concatenate(ptraj)
        list_of_ptrajs.append(ptraj)

    return list_of_ptrajs, np.array(which)

def load_trajectories(project, stride, atom_indices):

    list_of_trajs = []
    for i in xrange(project.n_trajs):
        # note, LoadTraj is only using the fast strided loading for
        # HDF5 formatted trajs
        traj = project.load_traj(i, stride=stride, atom_indices=atom_indices)
        
        if atom_indices != None:
            assert len(atom_indices) == traj.n_atoms
        
        list_of_trajs.append(traj)

    return list_of_trajs

    
def cluster(metric, trajectories, ptrajs, args, **kwargs):
    if args.alg == 'kcenters':
        clusterer = clustering.KCenters(metric, trajectories=trajectories,
            prep_trajectories=ptrajs, k=args.kcenters_num_clusters, 
            distance_cutoff=args.kcenters_distance_cutoff, seed=args.kcenters_seed)
    elif args.alg == 'hybrid':
        clusterer = clustering.HybridKMedoids(metric, trajectories=trajectories,
            prep_trajectories=ptrajs, k=args.hybrid_num_clusters,
            distance_cutoff=args.hybrid_distance_cutoff,
            local_num_iters=args.hybrid_local_num_iters,
            global_num_iters=args.hybrid_global_iters,
            too_close_cutoff=args.hybrid_too_close_cutoff,
            ignore_max_objective=args.hybrid_ignore_max_objective)
    elif args.alg == 'clarans':
        clusterer = clustering.Clarans(metric, trajectories=trajectories,
            prep_trajectories=ptrajs, 
            k=args.clarans_num_clusters,
            num_local_minima=args.clarans_num_local_minima,
            max_neighbors=args.clarans_max_neighbors,
            local_swap=args.clarans_local_swap)
    elif args.alg == 'sclarans':
        clusterer = clustering.SubsampledClarans(metric, trajectories=trajectories,
            prep_trajectories=ptrajs, 
            k=args.sclarans_num_clusters,
            num_samples=args.sclarans_num_samples,
            shrink_multiple=args.sclarans_shrink_multiple,
            num_local_minima=args.sclarans_num_local_minima,
            max_neighbors=args.sclarans_max_neighbors,
            local_swap=args.sclarans_local_swap,
            parallel=args.sclarans_parallel)
    elif args.alg == 'hierarchical':
        zmatrix_fn = kwargs['zmatrix_fn']
        clusterer = clustering.Hierarchical(metric, trajectories=trajectories,
            method=args.hierarchical_method)
        clusterer.save_to_disk(zmatrix_fn)
        logger.info('ZMatrix saved to %s. Use AssignHierarchical.py to assign the data', zmatrix_fn)
    else:
        raise ValueError('Unrecognized algorithm %s' % args.alg)
    
    return clusterer
    

def main(args, metric):
    
    if args.alg == 'sclarans' and args.stride != 1:
        logger.error("""You don't want to use a stride with sclarans. The whole point of
sclarans is to use a shrink multiple to accomplish the same purpose, but in parallel with
stochastic subsampling. If you cant fit all your frames into  memory at the same time, maybe you
could stride a little at the begining, but its not recommended.""")
        sys.exit(1)
        
    # if we have a metric that explicitly operates on a subset of indices,
    # then we provide the option to only load those indices into memory
    # WARNING: I also do something a bit dirty, and inject `None` for the
    # RMSD.atomindices to get the metric to not splice
    if isinstance(metric, metrics.RMSD):
        atom_indices = metric.atomindices
        metric.atomindices = None # probably bad...
        logger.info('RMSD metric - loading only the atom indices required')
    else:
        atom_indices = None

    # In case the clustering / algorithm needs extra arguments, use
    # this dictionary
    extra_kwargs = {}

    # Check to be sure we won't overwrite any data 
    if args.alg == 'hierarchical':
        zmatrix_fn = os.path.join(args.output_dir, 'ZMatrix.h5')
        die_if_path_exists(zmatrix_fn)
        extra_kwargs['zmatrix_fn'] = zmatrix_fn
    else:
        generators_fn = os.path.join(args.output_dir, 'Gens.h5') 
        die_if_path_exists(generators_fn)
        if args.stride == 1:
            assignments_fn = os.path.join(args.output_dir, 'Assignments.h5') 
            distances_fn = os.path.join(args.output_dir, 'Assignments.h5.distances')
            die_if_path_exists([assignments_fn, distances_fn])
        
    project = Project.load_from(args.project)

    if isinstance(metric, metrics.Vectorized) and not args.alg == 'hierarchical': 
        # if the metric is vectorized then
        # we can load prepared trajectories 
        # which may allow for better memory
        # efficiency
        ptrajs, which = load_prep_trajectories(project, args.stride, atom_indices, metric)
        trajectories = None
        n_trajs = len(ptrajs)

        num_frames = np.sum([len(p) for p in ptrajs])
        if num_frames != len(which):
            raise Exception("something went wrong in loading step (%d v %d)" % (num_frames, len(which)))
    else:
        trajectories = load_trajectories(project, args.stride, atom_indices)       
        ptrajs = None
        which = None
        n_trajs = len(trajectories)

    logger.info('Loaded %d trajs', n_trajs)

    clusterer = cluster(metric, trajectories, ptrajs, args, **extra_kwargs)

    if not isinstance(clusterer, clustering.Hierarchical):

        if isinstance(metric, metrics.Vectorized):
            gen_inds = clusterer.get_generator_indices()
            generators = project.load_frame(which[gen_inds,0], which[gen_inds,1])
        else:
            generators = clusterer.get_generators_as_traj()
        
        logger.info('Saving %s', generators_fn)
        generators.save(generators_fn)

        if args.stride == 1:
            assignments = clusterer.get_assignments()
            distances = clusterer.get_distances()
            
            logger.info('Since stride=1, Saving %s', assignments_fn)
            logger.info('Since stride=1, Saving %s', distances_fn)
            io.saveh(assignments_fn, assignments)
            io.saveh(distances_fn, distances)


def entry_point():
    args, metric = parser.parse_args()
    
    if hasattr(args, 'sclarans_parallel')  and args.sclarans_parallel == 'dtm':
        from deap import dtm
        dtm.start(main, args)
    else:
        main(args, metric)

if __name__ == "__main__":
    entry_point()
