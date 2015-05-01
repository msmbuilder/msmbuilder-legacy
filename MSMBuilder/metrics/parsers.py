from __future__ import print_function, division, absolute_import
import sys
import os
import pickle
import numpy as np
import mdtraj as md
import itertools
from pkg_resources import iter_entry_points
from msmbuilder.reduce import tICA
from msmbuilder.metrics import (RMSD, Dihedral, BooleanContact,
                                AtomPairs, ContinuousContact,
                                AbstractDistanceMetric, Vectorized,
                                RedDimPNorm, Positions, SolventFp)


def add_argument(group, *args, **kwargs):
    if 'default' in kwargs:
        d = 'Default: {d}'.format(d=kwargs['default'])
        if 'help' in kwargs:
            kwargs['help'] += ' {d}'.format(d=d)
        else:
            kwargs['help'] = d
    group.add_argument(*args, **kwargs)


def locate_metric_plugins(name):
    if not name in ['add_metric_parser', 'construct_metric', 'metric_class']:
        raise ValueError()

    eps = iter_entry_points(group='msmbuilder.metrics', name=name)
    return [ep.load() for ep in eps]


def add_metric_parsers(parser):

    metric_parser_list = []

    metric_subparser = parser.add_subparsers(dest='metric', description='Available metrics to use.')

    # metrics_parsers = parser.add_subparsers(description='Available metrics to use.',dest='metric')
    rmsd = metric_subparser.add_parser('rmsd',
                                       description='''RMSD: Root mean square deviation over a set of user defined atoms
        (typically backbone heavy atoms or alpha carbons). To evaluate the distance
        between two structures, first they are rotated and translated with respect
        to one another to achieve maximum coincidence. This code is executed in parallel
        on multiple cores (but not multiple boxes) using OMP.''')
    add_argument(rmsd, '-a', dest='rmsd_atom_indices', help='Atom indices to use in RMSD calculation. Pass "all" to use all atoms.',
                 default='AtomIndices.dat')
    metric_parser_list.append(rmsd)

    dihedral = metric_subparser.add_parser('dihedral',
                                           description='''DIHEDRAL: For each frame in the simulation data, we extract the
        torsion angles for the class of angles that you request (phi/psi is recommended,
        but chi angles are available as well). Each frame is then reprented by a vector
        containing the sin and cosine of these dihedral angles. The distances between
        frames are computed by taking distances between these vectors in R^n. The
        euclidean distance is recommended, but other distance metrics are available
        (cityblock, etc). This code is executed in parallel on multiple cores (but
        not multiple boxes) using OMP. ''')
    add_argument(dihedral, '-a', dest='dihedral_angles', default='phi/psi',
                 help='which dihedrals. Choose from phi, psi, chi (to choose multiple, seperate them with a slash), or user')
    add_argument(dihedral, '-f', dest='dihedral_userfilename', default='DihedralIndices.dat',
                 help='filename for dihedral indices, N lines of 4 space-separated indices (otherwise ignored)')
    add_argument(dihedral, '-p', dest='dihedral_p', default=2,
                 help='p used for metric=minkowski (otherwise ignored)')
    add_argument(dihedral, '-m', dest='dihedral_metric', default='euclidean',
                 help='which distance metric', choices=Dihedral.allowable_scipy_metrics)
    metric_parser_list.append(dihedral)

    contact = metric_subparser.add_parser('contact',
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
    add_argument(contact, '-C', dest='contact_cutoff', default=0.5,
                 help='Cutoff distance in nanometers. If you pass -1, then the contact "map" will be a matrix of residue-residue distances. Passing a number greater than 0 means the residue-residue distance matrix will be converted to a boolean matrix, one if the distance is less than the specified cutoff')
    add_argument(contact, '-f', dest='contact_cutoff_file',
                 help='File containing residue specific cutoff distances (supercedes the scalar cutoff distance if present).', default=None)
    add_argument(contact, '-s', dest='contact_scheme', default='closest-heavy', help='contact scheme.',
                 choices=['CA', 'closest', 'closest-heavy'])
    metric_parser_list.append(contact)

    atompairs = metric_subparser.add_parser('atompairs', description='''ATOMPAIRS: For each frame, we
        represent the conformation as a vector of particular atom-atom distances. Then the distance
        between frames is calculated using a specified norm on these vectors. This code is executed in
        parallel (but not multiple boxes) using OMP.''')
    add_argument(atompairs, '-a', dest='atompairs_which',
                 help='path to file with 2D array of which atompairs to use.', default='AtomPairs.dat')
    add_argument(atompairs, '-p', dest='atompairs_p', default=2,
                 help='p used for metric=minkowski (otherwise ignored)')
    add_argument(atompairs, '-m', dest='atompairs_metric', default='euclidean',
                 help='which distance metric', choices=AtomPairs.allowable_scipy_metrics)
    metric_parser_list.append(atompairs)

    solventfp = metric_subparser.add_parser('solventfp', description="""SOLVENTFP: For each frame
        calculate a 'solvent fingerprint' by considering the weighted pairwise distances
        between solute and solvent atoms as in as in Gu et al. BMC Bioinformatics 2013, 14(Suppl 2):S8.""")
    add_argument(solventfp, '-w', dest='solventfp_wateri', default='WaterIndices.dat',
                help='Indices of the solvent (water) atoms')
    add_argument(solventfp, '-v', dest='solventfp_proti', default='ProteinIndices.dat',
                help='Indices of the solute (protein) atoms')
    add_argument(solventfp, '-s', dest='solventfp_sigma', default=0.5,
                 type=float, help='std. dev. of gaussian kernel')
    add_argument(solventfp, '-p', dest='solventfp_p', default=2,
                 help='p used for metric=minkowski (otherwise ignored)')
    add_argument(solventfp, '-m', dest='solventfp_metric', default='euclidean',
        help='which distance metric', choices=SolventFp.allowable_scipy_metrics)
    metric_parser_list.append(solventfp)

    positions = metric_subparser.add_parser('positions', description="""POSITIONS: For each frame
        we represent the conformation as a vector of atom positions, where the atoms have been
        aligned to a target structure.""")
    add_argument(positions, '-t', dest='target',
                 help='target structure (PDB) to align structures to.')
    add_argument(positions, '-a', dest='pos_atom_indices',
                 help='atom indices to include in the distances.')
    add_argument(positions, '-i', dest='align_indices',
                 help='atom indices to use when aligning to target.')
    add_argument(positions, '-p', dest='positions_p', default=2,
                 help='p used for metric=minkowski (otherwise ignored)')
    add_argument(positions, '-m', dest='positions_metric', default='euclidean',
                 help='which distance metric', choices=Positions.allowable_scipy_metrics)
    metric_parser_list.append(positions)

    tica = metric_subparser.add_parser('tica', description='''
        tICA: This metric is based on a variation of PCA which looks for the slowest d.o.f.
        in the simulation data. See (Schwantes, C.R., Pande, V.S. JCTC 2013, 9 (4), 2000-09.)
        for more details. In addition to these options, you must provide an additional
        metric you used to prepare the trajectories in the training step.''')

    add_argument(tica, '-p', dest='p', help='p value for p-norm')
    add_argument(tica, '-m', dest='projected_metric', help='metric to use in the projected space',
                 choices=Vectorized.allowable_scipy_metrics, default='euclidean')
    add_argument(tica, '-f', dest='tica_fn',
                 help='tICA Object which was prepared by tICA_train.py')
    add_argument(tica, '-n', dest='num_vecs', type=int,
                 help='Choose the top <-n> eigenvectors based on their eigenvalues')
    metric_parser_list.append(tica)

    picklemetric = metric_subparser.add_parser('custom', description="""CUSTOM: Use a custom
        distance metric. This requires defining your metric and saving it to a file using
        the pickle format, which can be done fron an interactive shell. This is an EXPERT FEATURE,
        and requires significant knowledge of the source code's architecture to pull off.""")
    add_argument(picklemetric, '-i', dest='picklemetric_input', required=True,
                 help="Path to pickle file for the metric")
    metric_parser_list.append(picklemetric)

    for add_parser in locate_metric_plugins('add_metric_parser'):
        plugin_metric_parser = add_parser(metric_subparser, add_argument)
        metric_parser_list.append(plugin_metric_parser)

    return metric_parser_list

    ################################################################################


def construct_metric(args):
    metric_name = args.metric

    if metric_name == 'rmsd':
        if args.rmsd_atom_indices != 'all':
            atom_indices = np.loadtxt(args.rmsd_atom_indices, np.int)
        else:
            atom_indices = None
        metric = RMSD(atom_indices)  # , omp_parallel=args.rmsd_omp_parallel)

    elif metric_name == 'dihedral':
        metric = Dihedral(metric=args.dihedral_metric,
                          p=args.dihedral_p, angles=args.dihedral_angles,
                          userfilename=args.dihedral_userfilename)

    elif metric_name == 'contact':
        if args.contact_which != 'all':
            contact_which = np.loadtxt(args.contact_which, np.int)
        else:
            contact_which = 'all'

        if args.contact_cutoff_file != None:
            contact_cutoff = np.loadtxt(args.contact_cutoff_file, np.float)
        elif args.contact_cutoff != None:
            contact_cutoff = float(args.contact_cutoff)
        else:
            contact_cutoff = None

        if contact_cutoff != None and contact_cutoff < 0:
            metric = ContinuousContact(contacts=contact_which,
                                       scheme=args.contact_scheme)
        else:
            metric = BooleanContact(contacts=contact_which,
                                    cutoff=contact_cutoff, scheme=args.contact_scheme)

    elif metric_name == 'atompairs':
        if args.atompairs_which != None:
            pairs = np.loadtxt(args.atompairs_which, np.int)
        else:
            pairs = None

        metric = AtomPairs(metric=args.atompairs_metric, p=args.atompairs_p,
                           atom_pairs=pairs)

    elif metric_name == 'positions':
        target = md.load(args.target)

        if args.pos_atom_indices != None:
            atom_indices = np.loadtxt(args.pos_atom_indices, np.int)
        else:
            atom_indices = None

        if args.align_indices != None:
            align_indices = np.loadtxt(args.align_indices, np.int)
        else:
            align_indices = None

        metric = Positions(target, atom_indices=atom_indices, align_indices=align_indices,
                           metric=args.positions_metric, p=args.positions_p)

    elif metric_name == 'solventfp':
        proti = np.loadtxt(args.solventfp_proti, np.int)
        wateri = np.loadtxt(args.solventfp_wateri, np.int)
        metric = SolventFp(solute_indices=proti,
                           solvent_indices=wateri,
                           sigma=args.solventfp_sigma,
                           metric=args.solventfp_metric,
                           p=args.solventfp_p)

    elif metric_name == "tica":
        tica_obj = tICA.load(args.tica_fn)

        metric = RedDimPNorm(tica_obj, num_vecs=args.num_vecs,
                             metric=args.projected_metric, p=args.p)

    elif metric_name == 'custom':
        with open(args.picklemetric_input) as f:
            metric = pickle.load(f)
            print('#' * 80)
            print('Loaded custom metric:')
            print(metric)
            print('#' * 80)
    else:
        # apply the constructor on args and take the first non-none element
        # note that using these itertools constructs, we'll only actual
        # execute the constructor until the match is achieved
        metrics = [c(args) for c in locate_metric_plugins('construct_metric')]
        try:
            metric = next(itertools.dropwhile(lambda c: not c, metrics))
        except StopIteration:
            # This means that none of the plugins acceptedthe metric
            raise RuntimeError(
                "Bad metric. Could not be constructed by any built-in or plugin metric. Perhaps you have a poorly written plugin?")

    if not isinstance(metric, AbstractDistanceMetric):
        return ValueError("%s is not a AbstractDistanceMetric" % metric)

    return metric
