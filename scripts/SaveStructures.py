from __future__ import print_function, absolute_import, division
from mdtraj.utils.six.moves import xrange

import os
import numpy as np
from mdtraj import io
from msmbuilder import arglib
from msmbuilder import Project
from msmbuilder import MSMLib
from msmbuilder.clustering import concatenate_trajectories
import logging
logger = logging.getLogger('msmbuilder.scripts.SaveStructures')
DEBUG = True


parser = arglib.ArgumentParser(description="""
Yank a number of randomly selected conformations from each state in a model.

The conformations can either be saved in separate files (i.e. one PDB file per
conformations), or in the same file.
""")
parser.add_argument('project')
parser.add_argument('assignments', default='Data/Assignments.Fixed.h5')
parser.add_argument('conformations_per_state', default=5, type=int,
                    help='Number of conformations to sample from each state')
parser.add_argument('states', nargs='+', type=int,
                    help='''Which states to sample from. Pass a list of integers, separated
    by whitespace. To specify ALL of the states, include the integer -1.''',
                    default=[-1])
# nargs=+ means the values will come out as type list, but this isn't
# naturally applied to the default, so we just put the default as [-1]
parser.add_argument('format', choices=['pdb', 'xtc', 'h5'],
                    help='''Format for the outputted conformations. PDB is the standard
    plaintext protein databank format. XTC is the gromacs binary trajectory
    format, and h5 is the MDTraj hdf format''',
                    default='pdb')
parser.add_argument('style', choices=['sep', 'tps', 'one'], help='''Controls
    the number of conformations save per file. If "sep" (SEPARATE), all of
    the conformations will be saved in separate files, named in the format
    State{i}-{j}.{ext} for the `j`th conformation sampled from the `i`th
    state. If "tps" (Trajectory Per State), each of the conformations
    sampled from a given state `i` will be saved in a single file, named
    State{i}.{ext}. If "one", all of the conformations will be saved in a
    single file, such that the `j`th conformation from the `states[i]`-th
    microstate will be the `i+j*N`th frame in the trajectory file. The file
    will be namaed Confs.{ext}
    ''', default='sep')
parser.add_argument('replacement', type=bool, help='''Draw random stuctures from
    those assigned to a state either with replacement or without. If you ask for k
    structures from a state with < k assignments and are drawing with --replacement False,
    you will get an error.''', default=True)


def save(confs_by_state, states, style, format, outdir):
    "Save the results to disk"

    if style == 'sep':
        for i, trj in enumerate(confs_by_state):
            for j in xrange(len(trj)):

                fn = os.path.join(outdir, 'State%d-%d.%s' % (states[i], j,
                                                             format))
                arglib.die_if_path_exists(fn)

                logger.info("Saving file: %s" % fn)
                trj[j].save(fn)

    elif style == 'tps':
        for i, trj in enumerate(confs_by_state):
            fn = os.path.join(outdir, 'State%d.%s' % (states[i], format))
            arglib.die_if_path_exists(fn)

            logger.info("Saving file: %s" % fn)
            trj.save(fn)

    elif style == 'one':
        fn = os.path.join(outdir, 'Confs.%s' % format)
        arglib.die_if_path_exists(fn)

        logger.info("Saving file: %s" % fn)
        concatenate_trajectories(confs_by_state).save(fn)

    else:
        raise ValueError('Invalid style: %s' % style)


def entry_point():
    """Parse command line inputs, load up files, then call run() and save() to do
    the real work"""
    parser.add_argument('output_dir', default='PDBs')
    args = parser.parse_args()

    # load...
    # project
    project = Project.load_from(args.project)

    # assignments
    try:
        assignments = io.loadh(args.assignments, 'arr_0')
    except KeyError:
        assignments = io.loadh(args.assignments, 'Data')

    # states
    if -1 in args.states:
        states = np.unique(assignments[np.where(assignments != -1)])
        logger.info('Yanking from all %d states', len(states))
    else:
        # ensure that the states are sorted, and that they're unique -- you
        # can only request each state once
        states = np.unique(args.states)
        logger.info("Yanking from the following states: %s", states)

    # extract the conformations using np.random for the randomness
    confs_by_state = project.get_random_confs_from_states(
        assignments, states=states, num_confs=args.conformations_per_state,
        replacement=args.replacement)

    # save the conformations to disk, in the requested style
    save(confs_by_state=confs_by_state, states=states, style=args.style,
         format=args.format, outdir=args.output_dir)

if __name__ == '__main__':
    entry_point()
