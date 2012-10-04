import os
import numpy as np
import tables
import warnings
from msmbuilder import io
from msmbuilder import Trajectory
import logging
logger = logging.getLogger('assigning')

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
        io.saveh(filename, arr_0=-1*np.ones((project.n_trajs, np.max(project.traj_lengths)), dtype=dtype),
                 completed_trajs=np.zeros((project.n_trajs), dtype=np.bool))

    def check_container(filename):
        fh = tables.openFile(filename, 'r')
        if  fh.root.arr_0.shape != (project.n_trajs, np.max(project.traj_lengths)):
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
    f_assignments = tables.openFile(assignments_fn, mode='a')
    f_distances = tables.openFile(distances_fn, mode='a')

    return f_assignments, f_distances


def assign_in_memory(metric, generators, project):
    """This really should be called simple assign -- its the simplest"""

    n_trajs, max_traj_length = project.n_trajs, np.max(project.traj_lengths)
    assignments = -1 * np.ones((n_trajs, max_traj_length), dtype='int')
    distances = -1 * np.ones((n_trajs, max_traj_length), dtype='float32')

    pgens = metric.prepare_trajectory(generators)

    for i in xrange(n_trajs):
        traj = project.load_traj(i)
        ptraj = metric.prepare_trajectory(traj)
        for j in xrange(len(traj)):
            d = metric.one_to_all(ptraj, pgens, j)
            assignments[i,j] = np.argmin(d)
            distances[i,j] = d[assignments[i,j]]

    return assignments, distances


def assign_with_checkpoint(metric, project, generators, assignments_path, distances_path, chunk_size=10000):
    """
    Assign every frame to its closest generator
    
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
        
    Notes
    -----
    The results will be checkpointed along the way, trajectory by trajectory. So if
    the process is killed, it should be able to roughly pick up where it left off.
    """

    pgens = metric.prepare_trajectory(generators)
    
    # setup the file handles
    fh_a, fh_d = _setup_containers(project, assignments_path, distances_path)

    for i in xrange(project.n_trajs):
        if fh_a.root.completed_trajs[i] and fh_d.root.completed_trajs[i]:
            logger.info('Skipping trajectory %s -- already assigned', i)
            continue
        if fh_a.root.completed_trajs[i] or fh_d.root.completed_trajs[i]:
            raise RuntimeError('Corruption detected')
        logger.info('Assigning trajectory %s', i)
        
        # pointer to the position in the total trajectory where
        # the current chunk starts, so we know where in the Assignments
        # array to put each batch of data
        start_index = 0
        
        for tchunk in Trajectory.enum_chunks_from_lhdf(project.traj_filename(i), ChunkSize=chunk_size):
            ptchunk = metric.prepare_trajectory(tchunk)
            this_length = len(ptchunk)
            
            distances = np.empty(this_length, dtype=np.float32)
            assignments = np.empty(this_length, dtype=np.int)

            for j in xrange(this_length):
                d = metric.one_to_all(ptchunk, pgens, j)
                ind = np.argmin(d)
                assignments[j] = ind
                distances[j] = d[ind]
            
            end_index = start_index+this_length
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

def streaming_assign_with_checkpoint(metric, project, generators, assignments_path, distances_path, checkpoint=1,chunk_size=10000):
    warnings.warn("assign_with_checkpoint now uses the steaming engine -- this function is deprecated", DeprecationWarning)
    assign_with_checkpoint(metric, project, generators, assignments_path, distances_path, chunk_size)

