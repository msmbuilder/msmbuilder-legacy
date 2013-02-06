import os
import numpy as np
import tables

from parallel_assign.vtraj import VTraj
from msmbuilder import io

def partition(project, chunk_size):
    """Partition the frames in a project into a list of virtual trajectories
    (VTraj) of length <= chunk_size
    
    Returns
    -------
    vtrajs : list
        list of VTrajs
    """
    
    if not all(int(e) == e for e in project.traj_lengths):
        raise ValueError('must me ints')
    if not all(e > 0 for e in project.traj_lengths):
        raise ValueError('must be >0')
    
    def generate():
        "generator for the sections"
        last = VTraj(project)
        for i in xrange(0, project.n_trajs):
            end = 0
            if len(last) != 0:
                end = chunk_size - len(last)
                if  end < project.traj_lengths[i]:
                    last.append((i, 0, end))
                    yield last
                    last = VTraj(project)
                else:
                    end = project.traj_lengths[i]
                    last.append((i, 0, end))
                    
            for j in xrange(end, project.traj_lengths[i], chunk_size):
                if j + chunk_size <= project.traj_lengths[i]:
                    yield VTraj(project, (i, j, j + chunk_size))
                    last = VTraj(project)
                else:
                    last.append((i, j, project.traj_lengths[i]))
                
        if len(last) > 0:
            yield last
            
    all_vtrajs = list(generate())
    
    if sum(len(vt) for vt in all_vtrajs) != np.sum(project.traj_lengths):
        raise ValueError('Chunking error. Lengths dont match')
        
    return all_vtrajs


def setup_containers(outputdir, project, all_vtrajs):
    """
    Setup the files on disk (Assignments.h5 and Assignments.h5.distances) that
    results will be sent to.
    
    Check to ensure that if they exist (and contain partial results), the
    containers are not corrupted
    
    Parameters
    ----------
    outputdir : str
        path to save/find the files
    project : msmbuilder.Project
        The msmbuilder project file. Only the NumTrajs and TrajLengths are
        actully used (if you want to spoof it, you can just pass a dict)
    all_vtrajs : list
        The VTrajs are used to check that the containers on disk, if they
        exist, contain the right stuff
        
    Returns
    -------
    f_assignments : tables.File
        pytables handle to the assignments file, open in 'append' mode
    f_distances : tables.File
        pytables handle to the assignments file, open in 'append' mode    
    """
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    
    assignments_fn = os.path.join(outputdir, 'Assignments.h5')
    distances_fn = os.path.join(outputdir, 'Assignments.h5.distances')
    
    n_trajs = project.n_trajs
    max_n_frames = np.max(project.traj_lengths)
    n_vtrajs = len(all_vtrajs)
    hashes = np.array([c.hash() for c in all_vtrajs])
    
    minus_ones = -1 * np.ones((n_trajs, max_n_frames))
    
    def save_container(filename, dtype):
        io.saveh(filename, arr_0=np.array(minus_ones, dtype=dtype),
            completed_vtrajs=np.zeros((n_vtrajs), dtype=np.bool),
            hashes=hashes)
    
    def check_container(filename):
        ondisk = io.loadh(filename, deferred=False)
        if n_vtrajs != len(ondisk['hashes']):
            raise ValueError('You asked for {} vtrajs, but your checkpoint \
file has {}'.format(n_vtrajs, len(ondisk['hashes'])))
        if not np.all(ondisk['hashes'] ==
                hashes):
            raise ValueError('Hash mismatch. Are these checkpoint files for \
the right project?')
        
    
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
    

def save(f_assignments, f_distances, assignments, distances, vtraj):
    """
    Save assignments to disk
    
    Parameters
    ----------
    f_assignments : tables.File
        pytables handle to the assignments file to write to
    f_distances : tables.File
        pytables handle to the assignments file to write to
    assignments : np.ndarray
        1D array of the assignments for vtraj. should be dtype=np.int
    distanes : np.ndarray
        1D array of the distances for vtraj. should be dtype=np.float32
    vtraj : passign.VTraj
        logical trajectory object listing which physical trajectory/frames these
        assignments/distances correspond to
    """
    
    ptr = 0
    for trj_i, start, stop in vtraj:
        end = ptr + stop - start
            
        f_assignments.root.arr_0[trj_i, start:stop] = assignments[ptr:end]
        f_distances.root.arr_0[trj_i, start:stop] = distances[ptr:end]           
        ptr = end
    
    vtraj_i = np.where(f_assignments.root.hashes[:] == vtraj.hash())[0]
    if len(vtraj_i) != 1:
        raise ValueError('no matching vtraj?')
    vtraj_i = vtraj_i[0]
    
    f_assignments.root.completed_vtrajs[vtraj_i] = True
    f_assignments.flush()
    f_distances.flush()   
    
    return vtraj_i 
