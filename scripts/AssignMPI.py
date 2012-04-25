#!/usr/bin/env python
import sys, os
import numpy as np

import argparse
import pickle
from msmbuilder.Trajectory import Trajectory
from msmbuilder.assigning_mpi import MasterAssigner, WorkerAssigner
from msmbuilder.scripts.Cluster import add_argument, construct_metric
from msmbuilder import Project
from msmbuilder import metrics
from msmbuilder import mm

def main():
    if mm._RANK == 0:
        parser = argparse.ArgumentParser(description="""
Assign data that were not originally used in the clustering (because of
striding) to the microstates. This is applicable to all medoid-based clustering
algorithms, which includes all those implemented by Cluster.py except the
hierarchical methods. (For assigning to a hierarchical clustering, use
AssignHierarchical.py)

Outputs:
-Assignments.h5
-Assignments.h5.distances

Assignments.h5 contains the assignment of each frame of each trajectory to a
microstate in a rectangular array of ints. Assignments.h5.distances is an array
of real numbers of the same dimension containing the distance (according to
whichever metric you choose) from each frame to to the medoid of the
microstate it is assigned to.

This operation is performed for each trajectory in parallel using MPI, and can
be done accross multiple nodes in a cluster. Typically it is not advantageous
to use more than 1 MPI process per physical node, as the distance calculation
for most of the metrics (RMSD included) is done using shared memory parallelism
directly in C, and can thus fully leverage all of the cores on a single node.""",
 usage="""mpirun -np <num_concurrent> --bynode %s""" % os.path.split(sys.argv[0])[1], formatter_class=argparse.RawDescriptionHelpFormatter)
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
        
        picklemetric = metrics_parsers.add_parser('custom', description="""CUSTOM: Use a custom
        distance metric. This requires defining your metric and saving it to a file using
        the pickle format, which can be done fron an interactive shell. This is an EXPERT FEATURE,
        and requires significant knowledge of the source code's architecture to pull off.""")
        add_argument(picklemetric, '-i', dest='picklemetric_input', required=True,
            help="Path to pickle file for the metric")

        args = parser.parse_args()
        
        metric = construct_metric(args)
        assignments_path = os.path.join(args.output_dir, "Assignments.h5")
        distances_path = os.path.join(args.output_dir, "Assignments.h5.distances")
        lock_path = os.path.join(args.output_dir, "Assignments.lock")
        project = Project.LoadFromHDF(args.project)
        
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        
        # touch the lock_path file
        if not os.path.exists(lock_path):
            with open(lock_path, 'w') as f:
                pass
        
        master_kwargs = {'project': project, 'metric': metric, 'assignments_path': assignments_path,
                         'distances_path': distances_path, 'gensfn': args.generators,
                         'use_triangle': False, 'lock': lock_path}
    else:
        master_kwargs = {}
                        
    mm.start(MasterAssigner, WorkerAssigner, master_kwargs)
    
    os.unlink(lock_path)
    
if __name__ == '__main__':
    main()
