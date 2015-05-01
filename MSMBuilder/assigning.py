from __future__ import print_function, absolute_import, division
from mdtraj.utils.six.moves import xrange

import os
import mdtraj as md
import numpy as np
import tables
import warnings
from mdtraj import io
from msmbuilder import MSMLib as msmlib
import logging
logger = logging.getLogger(__name__)


def _setup_containers(project, assignments_fn, distances_fn):
    """
    Setup the files on disk (Assignments.h5 and Assignments.h5.distances) that
    results will be sent to.

    Check to ensure that if they exist (and contain partial results), the
    containers are not corrupted

    Parameters
    ----------
    project : msmbuilder.Project
        The msmbuilder project file. Only the n_trajs and traj_lengths are
        actully used.
    assignments_fn : string

    distances_fn : string

    Returns
    -------
    f_assignments : tables.File
        pytables handle to the assignments file, open in 'append' mode
    f_distances : tables.File
        pytables handle to the assignments file, open in 'append' mode
    """

    def save_container(filename, dtype):
        io.saveh(filename, arr_0=-1 * np.ones((project.n_trajs, np.max(project.traj_lengths)), dtype=dtype),
                 completed_trajs=np.zeros((project.n_trajs), dtype=np.bool))

    def check_container(filename):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fh = tables.openFile(filename, 'r')
        if fh.root.arr_0.shape != (project.n_trajs, np.max(project.traj_lengths)):
            raise ValueError('Shape error 1')
        if fh.root.completed_trajs.shape != (project.n_trajs,):
            raise ValueError('Shape error 2')
        fh.close()

    # save assignments container
    if (not os.path.exists(assignments_fn)) \
            and (not os.path.exists(distances_fn)):
        save_container(assignments_fn, np.int)
        save_container(distances_fn, np.float32)
    elif os.path.exists(assignments_fn) and os.path.exists(distances_fn):
        check_container(assignments_fn)
        check_container(distances_fn)
    else:
        raise ValueError("You're missing one of the containers")

    # append mode is read and write
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f_assignments = tables.openFile(assignments_fn, mode='a')
        f_distances = tables.openFile(distances_fn, mode='a')

    return f_assignments, f_distances


def assign_in_memory(metric, generators, project, atom_indices_to_load=None):
    """
    Assign every frame to its closest generator

    This code does everything in memory, and does not checkpoint. It also does
    not save any results to disk.

    Parameters
    ----------
    metric : msmbuilder.metrics.AbstractDistanceMetric
        A distance metric used to define "closest"
    project : msmbuilder.Project
        Used to load the trajectories
    generators : msmbuilder.Trajectory
        A trajectory containing the structures of all of the cluster centers
    atom_indices_to_load : {None, list}
        The indices of the atoms to load for each trajectory chunk. Note that
        this method is responsible for loading up atoms from the project, but
        does NOT load up the generators. Those are passed in as a trajectory
        object (above). So if the generators are already subsampled to a restricted
        set of atom indices, but the trajectories on disk are NOT, you'll
        need to pass in a set of indices here to resolve the difference.

    See Also
    --------
    assign_with_checkpoint
    """

    n_trajs, max_traj_length = project.n_trajs, np.max(project.traj_lengths)
    assignments = -1 * np.ones((n_trajs, max_traj_length), dtype='int')
    distances = -1 * np.ones((n_trajs, max_traj_length), dtype='float32')

    pgens = metric.prepare_trajectory(generators)

    for i in xrange(n_trajs):
        traj = project.load_traj(i, atom_indices=atom_indices_to_load)

        if traj['XYZList'].shape[1] != generators['XYZList'].shape[1]:
            raise ValueError('Number of atoms in generators does not match '
                             'traj we\'re trying to assign! Maybe check atom indices?')

        ptraj = metric.prepare_trajectory(traj)

        for j in xrange(len(traj)):
            d = metric.one_to_all(ptraj, pgens, j)
            assignments[i, j] = np.argmin(d)
            distances[i, j] = d[assignments[i, j]]

    return assignments, distances


def assign_with_checkpoint(metric, project, generators, assignments_path,
                           distances_path, chunk_size=10000, atom_indices_to_load=None):
    """
    Assign every frame to its closest generator

    The results will be checkpointed along the way, trajectory by trajectory.
    If the process is killed, it should be able to roughly pick up where it
    left off.

    Parameters
    ----------
    metric : msmbuilder.metrics.AbstractDistanceMetric
        A distance metric used to define "closest"
    project : msmbuilder.Project
        Used to load the trajectories
    generators : msmbuilder.Trajectory
        A trajectory containing the structures of all of the cluster centers
    assignments_path : str
        Path to a file that contains/will contain the assignments, as a 2D array
        of integers in hdf5 format
    distances_path : str
        Path to a file that contains/will contain the assignments, as a 2D array
        of integers in hdf5 format
    chunk_size : int
        The number of frames to load and process per step. The optimal number
        here depends on your system memory -- it should probably be roughly
        the number of frames you can fit in memory at any one time. Note, this
        is only important if your trajectories are long, as the effective chunk_size
        is really `min(traj_length, chunk_size)`
    atom_indices_to_load : {None, list}
        The indices of the atoms to load for each trajectory chunk. Note that
        this method is responsible for loading up atoms from the project, but
        does NOT load up the generators. Those are passed in as a trajectory
        object (above). So if the generators are already subsampled to a restricted
        set of atom indices, but the trajectories on disk are NOT, you'll
        need to pass in a set of indices here to resolve the difference.

    See Also
    --------
    assign_in_memory
    """

    pgens = metric.prepare_trajectory(generators)

    # setup the file handles
    fh_a, fh_d = _setup_containers(project, assignments_path, distances_path)

    for i in xrange(project.n_trajs):
        if fh_a.root.completed_trajs[i] and fh_d.root.completed_trajs[i]:
            logger.info('Skipping trajectory %s -- already assigned', i)
            continue
        if fh_a.root.completed_trajs[i] or fh_d.root.completed_trajs[i]:
            logger.warn("Re-assigning trajectory even though some data is"
                        " available...")
            fh_a.root.completed_trajs[i] = False
            fh_d.root.completed_trajs[i] = False
        logger.info('Assigning trajectory %s', i)

        # pointer to the position in the total trajectory where
        # the current chunk starts, so we know where in the Assignments
        # array to put each batch of data
        start_index = 0

        filename = project.traj_filename(i)
        chunkiter = md.iterload(filename, chunk=chunk_size, atom_indices=atom_indices_to_load)
        for tchunk in chunkiter:
            if tchunk.n_atoms != generators.n_atoms:
                msg = ("Number of atoms in generators does not match "
                       "traj we're trying to assign! Maybe check atom indices?")
                raise ValueError(msg)

            ptchunk = metric.prepare_trajectory(tchunk)

            this_length = len(ptchunk)

            distances = np.empty(this_length, dtype=np.float32)
            assignments = np.empty(this_length, dtype=np.int)

            for j in xrange(this_length):
                d = metric.one_to_all(ptchunk, pgens, j)
                ind = np.argmin(d)
                assignments[j] = ind
                distances[j] = d[ind]

            end_index = start_index + this_length
            fh_a.root.arr_0[i, start_index:end_index] = assignments
            fh_d.root.arr_0[i, start_index:end_index] = distances

            # i'm not sure exactly what the optimal flush frequency is
            fh_a.flush()
            fh_d.flush()
            start_index = end_index

        # we're going to keep duplicates of this record -- i.e. writing
        # it to both files

        # completed chunks are not checkpointed -- only completed trajectories
        # this means that if the process dies after completing 10/20 of the
        # chunks in trajectory i -- those chunks are going to have to be recomputed
        # (put trajectory i-1 is saved)

        # this could be changed, but the implementation is a little tricky -- you
        # have to watch out for the fact that the person might call this function
        # with chunk_size=N, let it run for a while, kill it, and then call it
        # again with chunk_size != N. Dealing with that appropriately is tricky
        # since the chunks wont line up in the two cases
        fh_a.root.completed_trajs[i] = True
        fh_d.root.completed_trajs[i] = True

    fh_a.close()
    fh_d.close()


def streaming_assign_with_checkpoint(metric, project, generators, assignments_path,
                                     distances_path, checkpoint=1, chunk_size=10000, atom_indices_to_load=None):
    warnings.warn(("assign_with_checkpoint now uses the steaming engine "
                   "-- this function is deprecated"), DeprecationWarning)
    assign_with_checkpoint(metric, project, generators, assignments_path,
                           distances_path, chunk_size, atom_indices_to_load)


def outer_product_assignment(assign1, assign2):
    """Combine the results of two clusterings.

    Specifically, if a conformation is in state i after clustering
    with metric 1, and it is in state j after clustering with metric 2,
    we assign it to a new state (ij)

    This is a naive way of combining two metrics to give a singular
    assignment of states.

    Parameters
    ----------
    assign1 : np.ndarray
        Assignment array 1
    assign2 : np.ndarray
        Assignment array 2

    Returns
    ----------
    new_assign : np.ndarray
        New assignments array, renumbered from 0 ... N-1, N <= n1 * n2
    """

    assert assign1.shape == assign2.shape, """Assignments must be
        for the same set of trajectories."""
    new_assign = -1 * np.ones_like(assign1, dtype=np.int)

    nstates1 = np.max(assign1) + 1
    nstates2 = np.max(assign2) + 1

    translations = np.reshape(np.arange(nstates1 * nstates2),
                              (nstates1, nstates2))

    ass_shape = assign1.shape
    for i in xrange(ass_shape[0]):
        for j in xrange(ass_shape[1]):
            # TODO: vectorize this loop
            if assign1[i, j] == -1:
                # No assignment here
                assert assign2[i, j] == -1, """Assignments must be for
                    the same set of trajectories."""
            else:
                new_assign[i, j] = translations[assign1[i, j], assign2[i, j]]

    # Renumber states in place
    msmlib.renumber_states(new_assign)
    return new_assign

