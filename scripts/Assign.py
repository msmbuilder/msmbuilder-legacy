#!/usr/bin/env python
import sys, os
import numpy as np

#from Emsmbuilder import arglib
import argparse
from Emsmbuilder.Trajectory import Trajectory
from Emsmbuilder.scripts.Cluster import add_argument, construct_metric
from Emsmbuilder.assigning import assign_with_checkpoint
from Emsmbuilder import metrics
from Emsmbuilder import Project

def main():
    parser = argparse.ArgumentParser()
    add_argument(parser, '-p', dest='project', help='Path to ProjectInfo file',
        default='ProjectInfo.h5')
    add_argument(parser, '-g', dest='generators', help='''Output trajectory file containing
        the structures of each of the cluster centers. Note that for hierarchical clustering
        methods, this file will not be produced.''', default='Data/Gens.lh5')
    add_argument(parser, '-o', dest='output_dir', help='Location to save results', default='Data/')
    
    metrics_parsers = parser.add_subparsers(dest='metric')
    rmsd = metrics_parsers.add_parser('rmsd',
        description='''RMSD: Root mean square deviation over a set of user defined atoms
        (typically backbone heavy atoms or alpha carbons). To evaluate the distance
        between two structures, first they are rotated and translated with respect
        to one another to achieve maximum coincidence. This code is executed in parallel
        on multiple cores (but not multiple boxes) using OMP. You may choose from the
        following clustering algorithms:''')
    add_argument(rmsd, '-a', dest='rmsd_atom_indices', help='atomindices', default='AtomIndices.dat')

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
    more than one set of indistinguishable atoms''')

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
    args = parser.parse_args()
    
    metric = construct_metric(args)
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
