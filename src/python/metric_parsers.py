#!/usr/bin/env python

import sys, os
import pickle
from msmbuilder import metrics
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

def add_metric_parsers(parser):

    metrics_parsers = parser.add_subparsers(description='Available metrics to use.',dest='metric')
    rmsd = metrics_parsers.add_parser('rmsd',
        description='''RMSD: Root mean square deviation over a set of user defined atoms
        (typically backbone heavy atoms or alpha carbons). To evaluate the distance
        between two structures, first they are rotated and translated with respect
        to one another to achieve maximum coincidence. This code is executed in parallel
        on multiple cores (but not multiple boxes) using OMP.''')
    add_argument(rmsd, '-a', dest='rmsd_atom_indices', help='Atom indices to use in RMSD calculation. Pass "all" to use all atoms.', 
        default='AtomIndices.dat')
    #rmsd_subparsers = rmsd.add_subparsers()
    #rmsd_subparsers.metric = 'rmsd'

    dihedral = metrics_parsers.add_parser('dihedral',
        description='''DIHEDRAL: For each frame in the simulation data, we extract the
        torsion angles for the class of angles that you request (phi/psi is recommended,
        but chi angles are available as well). Each frame is then reprented by a vector
        containing the sin and cosine of these dihedral angles. The distances between
        frames are computed by taking distances between these vectors in R^n. The
        euclidean distance is recommended, but other distance metrics are available
        (cityblock, etc). This code is executed in parallel on multiple cores (but
        not multiple boxes) using OMP. ''') 
    add_argument(dihedral, '-a', dest='dihedral_angles', default='phi/psi',
        help='which dihedrals. Choose from phi, psi, chi. To choose multiple, seperate them with a slash')
    add_argument(dihedral, '-p', dest='dihedral_p', default=2, help='p used for metric=minkowski (otherwise ignored)')
    add_argument(dihedral, '-m', dest='dihedral_metric', default='euclidean',
        help='which distance metric', choices=metrics.Dihedral.allowable_scipy_metrics)
#    dihedral_subparsers = dihedral.add_subparsers()
#    dihedral_subparsers.metric = 'dihedral'

    lprmsd = metrics_parsers.add_parser('lprmsd',
        description='''LPRMSD: RMSD with the ability to to handle permutation-invariant atoms.
        Solves the assignment problem using a linear programming solution (LP). Can handle aligning
        on some atoms and computing the RMSD on other atoms.:''')
    add_argument(lprmsd, '-a', dest='lprmsd_atom_indices', help='Regular atom indices. Pass "all" to use all atoms.', default='AtomIndices.dat')
    add_argument(lprmsd, '-l', dest='lprmsd_alt_indices', default=None,
        help='''Optional alternate atom indices for RMSD. If you want to align the trajectories
        using one set of atom indices but then compute the distance using a different
        set of indices, use this option. If supplied, the regular atom_indices will
        be used for the alignment and these indices for the distance calculation''')
    add_argument(lprmsd, '-P', dest='lprmsd_permute_atoms', default=None, help='''Atom labels to be permuted.
    Sets of indistinguishable atoms that can be permuted to minimize the RMSD. On disk this should be stored as
    a list of newline separated indices with a "--" separating the sets of indices if there are
    more than one set of indistinguishable atoms.  Use "-- (integer)" to include a subset in the RMSD (to avoid undesirable boundary effects.)''')
#    lprmsd_subparsers = lprmsd.add_subparsers()
#    lprmsd_subparsers.metric = 'lprmsd'

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
        not multiple boxes) using OMP.''')
    add_argument(contact, '-c', dest='contact_which', default='all',
        help='Path to file containing 2D array of the contacts you want, or the string "all".')
    add_argument(contact, '-C', dest='contact_cutoff', default=0.5, help='Cutoff distance in nanometers.')
    add_argument(contact, '-f', dest='contact_cutoff_file', help='File containing residue specific cutoff distances (supercedes the scalar cutoff distance if present).',default=None)
    add_argument(contact, '-s', dest='contact_scheme', default='closest-heavy', help='contact scheme.',
        choices=['CA', 'closest', 'closest-heavy'])
    #contact_subparsers = contact.add_subparsers()
    #contact_subparsers.metric = 'contact'
    
    atompairs = metrics_parsers.add_parser('atompairs',description='''ATOMPAIRS: For each frame, we
        represent the conformation as a vector of particular atom-atom distances. Then the distance
        between frames is calculated using a specified norm on these vectors. This code is executed in
        parallel (but not multiple boxes) using OMP.''') 
    add_argument(atompairs, '-a', dest='atompairs_which',
        help='path to file with 2D array of which atompairs to use.', default='AtomPairs.dat')
    add_argument(atompairs, '-p', dest='atompairs_p', default=2, help='p used for metric=minkowski (otherwise ignored)')
    add_argument(atompairs, '-m', dest='atompairs_metric', default='cityblock',
        help='which distance metric', choices=metrics.AtomPairs.allowable_scipy_metrics)
    #atompairs_subparsers = atompairs.add_subparsers()
    #atompairs_subparsers.metric = 'atompairs'
    
    picklemetric = metrics_parsers.add_parser('custom', description="""CUSTOM: Use a custom
        distance metric. This requires defining your metric and saving it to a file using
        the pickle format, which can be done fron an interactive shell. This is an EXPERT FEATURE,
        and requires significant knowledge of the source code's architecture to pull off.""")
    add_argument(picklemetric, '-i', dest='picklemetric_input', required=True,
        help="Path to pickle file for the metric")
#    picklemetric_subparsers = picklemetric.add_subparsers()
#    picklemetric_subparsers.metric = 'custom'
    
    ################################################################################
    
def construct_metric(args):
    if args.metric == 'rmsd':
        if args.rmsd_atom_indices != 'all':
            atom_indices = np.loadtxt(args.rmsd_atom_indices, np.int)
        else:
            atom_indices = None
        metric = metrics.RMSD(atom_indices)#, omp_parallel=args.rmsd_omp_parallel)

    elif args.metric == 'dihedral':
        metric = metrics.Dihedral(metric=args.dihedral_metric,
            p=args.dihedral_p, angles=args.dihedral_angles)
             
    elif args.metric == 'lprmsd':
        from msmbuilder.metric_LPRMSD import LPRMSD, LPTraj, ReadPermFile

        if args.lprmsd_atom_indices != 'all':
            atom_inds = np.loadtxt( args.lprmsd_atom_indices )
        else:
            atom_inds = None

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
            contact_which = np.loadtxt(args.contact_which,np.int)
        else:
            contact_which = 'all'

        if args.contact_cutoff_file != None: #getattr(args, 'contact_cutoff_file'):
            contact_cutoff = np.loadtxt(args.contact_cutoff_file, np.float)            
        else:
            contact_cutoff = None
             
        metric = metrics.BooleanContact(contacts=contact_which,
            cutoff=contact_cutoff, scheme=args.contact_scheme)
     
    elif args.metric == 'atompairs':
        if args.atompairs_which != None:
            pairs = np.loadtxt(args.atompairs_which, np.int)
        else:
            pairs = None

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
