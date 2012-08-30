import sys, os
import numpy as np
from msmbuilder.Serializer import Serializer
from msmbuilder.Trajectory import Trajectory

def _setup_containers(assignments_path, distances_path, num_trajs, longest):
    "helper method"
    assignments_tmp = assignments_path + '.tmp'
    distances_tmp = distances_path + '.tmp'
    if os.path.exists(assignments_tmp):
        os.remove(assignments_tmp)
    if os.path.exists(distances_tmp):
        os.remove(distances_tmp)
    
    all_distances = -1 * np.ones((num_trajs, longest), dtype=np.float32)
    all_assignments = -1 * np.ones((num_trajs, longest), dtype=np.int)
    
    if os.path.exists(assignments_path) and os.path.exists(distances_path):
        s = Serializer.LoadFromHDF(assignments_path)
        t = Serializer.LoadFromHDF(distances_path)
        all_distances = s['Data']
        all_assignments = t['Data']
        
        try:
            completed_trajectories = s['completed_trajectories']
        except:
            completed_trajectories = (all_assignments[:, 0] >= 0)
        
    else:
        print "Creating serializer containers"
        completed_trajectories = np.array([False] * num_trajs)
        Serializer({'Data': all_assignments,
                    'completed_trajectories': completed_trajectories
                    }).SaveToHDF(assignments_path)
        Serializer({'Data': all_distances}).SaveToHDF(distances_path)
        
    return assignments_tmp, distances_tmp, all_assignments, all_distances, completed_trajectories


def assign_in_memory(metric, generators, project):
    """This really should be called simple assign -- its the simplest"""

    n_trajs, max_traj_length = project['NumTrajs'], max(project['TrajLengths'])
    assignments = -1 * np.ones((n_trajs, max_traj_length), dtype='int')
    distances = -1 * np.ones((n_trajs, max_traj_length), dtype='float32')

    pgens = metric.prepare_trajectory(generators)
    
    for i in xrange(n_trajs):
        traj = project.LoadTraj(i)
        ptraj = metric.prepare_trajectory(traj)
        for j in xrange(len(traj)):
            d = metric.one_to_all(ptraj, pgens, j)
            assignments[i, j] = np.argmin(d)
            distances[i, j] = d[assignments[i, j]]

    return assignments, distances
    

def assign_with_checkpoint(metric, project, generators, assignments_path, distances_path, checkpoint=1):
    """Assign each of the frames in each of the trajectories in the supplied project to
    their closest generator (frames of the trajectory "generators") using the supplied
    distance metric.
    
    assignments_path and distances_path should be the filenames of to h5 files in which the
    results are/will be stored. The results are stored along the way as they are computed,
    and if this method is killed halfway through execution, you can restart it and it
    will not have lost its place (i.e. checkpointing)"""
    pgens = metric.prepare_trajectory(generators)
    
    num_trajs = project['NumTrajs']
    longest = max(project['TrajLengths'])
    
    assignments_tmp, distances_tmp, all_assignments, all_distances, completed_trajectories = _setup_containers(assignments_path,
        distances_path, num_trajs, longest)
    
    for i in xrange(project['NumTrajs']):
        if completed_trajectories[i]:
            print 'Skipping trajectory %d -- already assigned' % i
            continue
        print 'Assigning trajectory %d' % i
        
        ptraj = metric.prepare_trajectory(project.LoadTraj(i))
        distances = -1 * np.ones(len(ptraj), dtype=np.float32)
        assignments = -1 * np.ones(len(ptraj), dtype=np.int)
        
        for j in range(len(ptraj)):
            d = metric.one_to_all(ptraj, pgens, j)
            ind = np.argmin(d)
            assignments[j] = ind
            distances[j] = d[ind]
            
        all_distances[i, 0:len(ptraj)] = distances
        all_assignments[i, 0:len(ptraj)] = assignments
        completed_trajectories[i] = True
        # checkpoint every few trajectories
        if ((i+1) % checkpoint == 0) or (i+1 == project['NumTrajs']):
            Serializer({'Data': all_assignments,
                        'completed_trajectories': completed_trajectories
                        }).SaveToHDF(assignments_tmp)
            Serializer({'Data': all_distances}).SaveToHDF(distances_tmp)
            os.rename(assignments_tmp, assignments_path)
            os.rename(distances_tmp, distances_path)

    return all_assignments, all_distances
