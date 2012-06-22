#!/usr/bin/env python

import sys, os
import pickle
from pprint import pprint
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

parser = argparse.ArgumentParser()
add_argument(parser, '-p', dest='project', help='Path to ProjectInfo file',
    default='ProjectInfo.h5')
add_argument(parser, '-S', dest='stride', help='Subsample by striding',
    default=1, type=int)
add_argument(parser, '-a', dest='assignments', help='''Output assignments file
    (will be used if stride is 1 and you're not using hierarchical)''',
    default='Data/Assignments.h5')
add_argument(parser, '-d', dest='distances', help='''Output assignments distances file.
    Will be used if stride is 1 and you're not using hierarchical.
    (distance from each data point to its cluster center according to your selected
    distance metric). Note that for hierarchical clustering methods, this file will
    not be produced.''', default='Data/Assignments.h5.distances')
add_argument(parser, '-g', dest='generators', help='''Output trajectory file containing
    the structures of each of the cluster centers. Note that for hierarchical clustering
    methods, this file will not be produced.''', default='Data/Gens.lh5')

metrics_parsers = parser.add_subparsers()
rmsd = metrics_parsers.add_parser('rmsd',
    description='''RMSD: Root mean square deviation over a set of user defined atoms
    (typically backbone heavy atoms or alpha carbons). To evaluate the distance
    between two structures, first they are rotated and translated with respect
    to one another to achieve maximum coincidence. This code is executed in parallel
    on multiple cores (but not multiple boxes) using OMP. You may choose from the
    following clustering algorithms:''')
add_argument(rmsd, '-a', dest='rmsd_atom_indices', help='atomindices', default='AtomIndices.dat')
rmsd_subparsers = rmsd.add_subparsers()
rmsd_subparsers.metric = 'rmsd'

dihedral = metrics_parsers.add_parser('dihedral',
    description='''DIHEDRAL: For each frame in the simulation data, we extract the
    torsion angles for the class of angles that you request (phi/psi is recommended,
    but chi angles are available as well). Each frame is then reprented by a vector
    containing the sin and cosine of these dihedral angles. The distances between
    frames are computed by taking distances between these vectors in R^n. The
    euclidean distance is recommended, but other distance metrics are available
    (cityblock, etc). This code is executed in parallel on multiple cores (but
    not multiple boxes) using OMP. You may choose from the following clustering algorithms:''') 
add_argument(dihedral, '-a', dest='dihedral_angles', default='phi/psi',
    help='which dihedrals. Choose from phi, psi, chi. To choose multiple, seperate them with a slash')
add_argument(dihedral, '-p', dest='dihedral_p', default=2, help='p used for metric=minkowski (otherwise ignored)')
add_argument(dihedral, '-m', dest='dihedral_metric', default='euclidean',
    help='which distance metric', choices=metrics.Dihedral.allowable_scipy_metrics)
dihedral_subparsers = dihedral.add_subparsers()
dihedral_subparsers.metric = 'dihedral'

lprmsd = metrics_parsers.add_parser('lprmsd',
    description='''LPRMSD: RMSD with the ability to to handle permutation-invariant atoms.
Solves the assignment problem using a linear programming solution (LP). Can handle aligning
on some atoms and computing the RMSD on other atoms. You may choose from the following clustering algorithms:''')
add_argument(lprmsd, '-a', dest='lprmsd_atom_indices', help='Regular atom indices', default='AtomIndices.dat')
add_argument(lprmsd, '-l', dest='lprmsd_alt_indices', default=None,
    help='''Optional alternate atom indices for RMSD. If you want to align the trajectories
    using one set of atom indices but then compute the distance using a different
    set of indices, use this option. If supplied, the regular atom_indices will
    be used for the alignment and these indices for the distance calculation''')
add_argument(lprmsd, '-P', dest='lprmsd_permute_atoms', default=None, help='''Atom labels to be permuted.
Sets of indistinguishable atoms that can be permuted to minimize the RMSD. On disk this should be stored as
a list of newline separated indices with a "--" separating the sets of indices if there are
more than one set of indistinguishable atoms.  Use "-- (integer)" to include a subset in the RMSD (to avoid undesirable boundary effects.)''')
lprmsd_subparsers = lprmsd.add_subparsers()
lprmsd_subparsers.metric = 'lprmsd'

contact = metrics_parsers.add_parser('contact',
    description='''CONTACT: For each frame in the simulation data, we extract the
contact map (presence or absense of "contacts")  between residues. Each frame is then
represented as a boolean valued vector containing information between the presence or
absense of certain contacts. The contact vector can either include all possible pairwise
contacts, only the native contacts, or any other set of pairs of residues. The distance with
which two residues must be within to classify as "in contact" is also settable, and can
dependend on the contact (e.g. 5 angstroms from some pairs, 10 angstroms for other pairs).
Furthermore, the sense in which the distance between two residues is computed can be
either specified as "CA", "closest", or "closest-heavy", which will respectively compute
("CA") the distance between the residues' alpha carbons, ("closest"), the closest distance between any pair of
atoms i and j such that i belongs to one residue and j to the other residue, ("closest-heavy"), 
or the closest distance between any pair of NON-HYDROGEN atoms i and j such that i belongs to
one residue and j to the other residue. This code is executed in parallel on multiple cores (but
not multiple boxes) using OMP. You may choose from the following clustering algorithms:''')
add_argument(contact, '-c', dest='contact_which', default='all',
    help='Path to file containing 2D array of the contacts you want, or the string "all".')
add_argument(contact, '-C', dest='contact_cutoff', default=0.5, help='Cutoff distance in nanometers.')
add_argument(contact, '-f', dest='contact_cutoff_file', help='File containing residue specific cutoff distances (supercedes the scalar cutoff distance if present).')
add_argument(contact, '-s', dest='contact_scheme', default='closest-heavy', help='contact scheme.',
    choices=['CA', 'cloest', 'closest-heavy'])
contact_subparsers = contact.add_subparsers()
contact_subparsers.metric = 'contact'

atompairs = metrics_parsers.add_parser('atompairs') 
add_argument(atompairs, '-a', dest='atompairs_which',
    help='path to file with 2D array of which atompairs to use.', default='AtomPairs.dat')
add_argument(atompairs, '-p', dest='atompairs_p', default=2, help='p used for metric=minkowski (otherwise ignored)')
add_argument(atompairs, '-m', dest='atompairs_metric', default='cityblock',
    help='which distance metric', choices=metrics.AtomPairs.allowable_scipy_metrics)
atompairs_subparsers = atompairs.add_subparsers()
atompairs_subparsers.metric = 'atompairs'

picklemetric = metrics_parsers.add_parser('custom', description="""CUSTOM: Use a custom
distance metric. This requires defining your metric and saving it to a file using
the pickle format, which can be done fron an interactive shell. This is an EXPERT FEATURE,
and requires significant knowledge of the source code's architecture to pull off.""")
add_argument(picklemetric, '-i', dest='picklemetric_input', required=True,
    help="Path to pickle file for the metric")
picklemetric_subparsers = picklemetric.add_subparsers()
picklemetric_subparsers.metric = 'custom'

################################################################################

subparsers = [rmsd_subparsers, dihedral_subparsers, lprmsd_subparsers, contact_subparsers, atompairs_subparsers, picklemetric_subparsers]
for subparser in subparsers:
    kcenters = subparser.add_parser('kcenters') 
    kcenters.set_defaults(alg='kcenters', metric=subparser.metric)
    add_argument(kcenters, '-s', help='seed for initial cluster center.', default=0, type=int, dest='kcenters_seed')
    kcenters_cutoff = kcenters.add_argument_group('cutoff (use one)').add_mutually_exclusive_group(required=True)
    add_argument(kcenters_cutoff, '-k', help='number of clusters',
        type=int, dest='kcenters_num_clusters')
    add_argument(kcenters_cutoff, '-d', help='no greater cophenetic distance than this cutoff',
        type=float, dest='kcenters_distance_cutoff')
    
    
    hybrid = subparser.add_parser('hybrid')
    hybrid.set_defaults(alg='hybrid', metric=subparser.metric)
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
    clarans.set_defaults(alg='clarans', metric=subparser.metric)
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
    sclarans.set_defaults(alg='sclarans', metric=subparser.metric)
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
    hier.set_defaults(alg='hierarchical', metric=subparser.metric)
    add_argument(hier, '-m', default='ward', help='method. default=ward',
        choices=['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'], dest='hierarchical_method')
    add_argument(hier, '-o', dest='hierarchical_save_zmatrix', help='Save Z-matrix to disk', default='Data/Zmatrix.h5')

def construct_metric(args):
    if args.metric == 'rmsd':
        atom_indices = np.loadtxt(args.rmsd_atom_indices, np.int)
        metric = metrics.RMSD(atom_indices)#, omp_parallel=args.rmsd_omp_parallel)

    elif args.metric == 'dihedral':
        metric = metrics.Dihedral(metric=args.dihedral_metric,
            p=args.dihedral_p, angles=args.dihedral_angles)
            
    elif args.metric == 'lprmsd':
        from msmbuilder.metric_LPRMSD import LPRMSD, LPTraj, ReadPermFile
        atom_inds = np.loadtxt(args.lprmsd_atom_indices, np.int)
        if args.lprmsd_permute_atoms is not None:
            permute_inds = ReadPermFile(args.lprmsd_permute_atoms)
        else:
            permute_inds = None
        if args.lprmsd_alt_indices is not None:
            alt_inds = np.loadtxt(args.lprmsd_alt_indices, np.int)
        else:
            alt_inds = None

        metric = LPRMSD(atom_inds, permute_inds, alt_inds)
        
    elif args.metric == 'contact':
        if args.contact_which != 'all':
            args.contact_which = np.loadtxt(args.contact_which, np.int)
        if getattr(args, 'contact_cutoff_file'):
            args.contact_cutoff = np.loadtxt(args.contact_cutoff_file, np.float)            
            
        metric = metrics.BooleanContact(contacts=args.contact_which,
            cutoff=args.contact_cutoff, scheme=args.contact_scheme)
    
    elif args.metric == 'atompairs':
        pairs = np.loadtxt(args.atompairs_which, np.int)
        metric = metrics.AtomPairs(metric=args.atompairs_metric, p=args.atompairs_p,
            atom_pairs=pairs)
            
    elif args.metric == 'custom':
        with open(args.picklemetric_input) as f:
            metric = pickle.load(f)
            print '#'*80
            print 'Loaded custom metric:'
            print metric
            print '#'*80
    else:
        raise Exception("Bad metric")
    
    return metric

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

    
def main(args):
    check_paths(args)
    
    if args.alg == 'sclarans' and args.stride != 1:
        print >> sys.stderr, """\nYou don't want to use a stride with sclarans. The whole point of
sclarans is to use a shrink multiple to accomplish the same purpose, but in parallel with
stochastic subsampling. If you cant fit all your frames into  memory at the same time, maybe you
could stride a little at the begining, but its not recommended."""
        sys.exit(1)
    
    metric = construct_metric(args)

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
    print LicenseString
    print CiteString
    print ''
    print 'Cluster.py: Cluster MD trajectories into microstates'
    print 
    print 'Output: Assignments.h5, and other files depending on your choice of distance'
    print 'metric and/or clustering algorithm.'
    print 
    print 'Note that %d distance metrics and %d clustering algorithms are available' % (len(subparsers), 5)
    print 'Many of which have multiple options and parameters.'
    print 
    print 'MAKE LIBERAL USE OF THE -h OPTION. The help text changes significantly'
    print 'depending on which level in the options tree you are currently in'
    
    print '\n' + '-' * 80
    args = parser.parse_args()
    pprint(args.__dict__)
    
    if hasattr(args, 'sclarans_parallel')  and args.sclarans_parallel == 'dtm':
        from deap import dtm
        dtm.start(main, args)
    else:
        main(args)


    
    
    
        
