import sys, os
import numpy as np
from msmbuilder import mm
from msmbuilder.Serializer import Serializer 
from msmbuilder.Trajectory import Trajectory
from msmbuilder.clustering import _clarans, empty_trajectory_like
from msmbuilder.utils import uneven_zip
from collections import deque
from msmbuilder.assigning import _setup_containers

try:
    import portalocker
except:
    print >> sys.stderr, 'In order to use MPI assigning, please install'
    print >> sys.stderr, 'the portalocker package. It can be installed using'
    print >> sys.stderr, '"easy_install portalocker"'
    sys.exit(1)


class MasterAssigner(mm.BaseMaster):
    def run(self):
        
        assignments_tmp, distances_tmp, a0, a1, completed_trajectories = _setup_containers(self.assignments_path,
            self.distances_path, self.project['NumTrajs'], max(self.project['TrajLengths']))
        del a0, a1
        
        if self.use_triangle:
            method = 'assign_triangle'
            self._setup_triangle()
            raise NotImplementedError('Sorry!')
        else:
            method = 'assign_notriangle'
            for w in self.workers:
                self.send(w, 'setup', self.project, self.metric, self.gensfn, assignments_tmp, distances_tmp,
                                      self.assignments_path, self.distances_path, self.lock)
        
        self.pgens = self.metric.prepare_trajectory(Trajectory.LoadTrajectoryFile(self.gensfn))
        trajs_to_run = []
        for i,t in enumerate(completed_trajectories):
            if t:
                self.log('Skipping trajectory %d -- already assigned' % i)
            else:
                trajs_to_run.append(i)
            
        if len(trajs_to_run) > 0:
            self.map(method, trajs_to_run)
        
        self.log('Finished')
        
    def _setup_triangle(self):
        metagens_k = 24
        metagens_i = _clarans(self.metric, self.pgens, metagens_k, num_local_minima=5, max_neighbors=5, local_swap=True, verbose=True)[0]
        self.tfilter = TriangleFilter(self.metric, metagens_i, self.gens)
        self.metagens = empty_trajectory_like(self.gens)
        self.metagens['XYZList'] = self.gens['XYZList'][metagens_i]

class WorkerAssigner(mm.BaseWorker):
    def setup(self, project, metric, gensfn, assignments_tmp, distances_tmp, assignments_path, distances_path, lock):
        self.project = project
        self.metric = metric
        self.pgens = metric.prepare_trajectory(Trajectory.LoadTrajectoryFile(gensfn))
        self.assignments_tmp = assignments_tmp
        self.distances_tmp = distances_tmp
        self.assignments_path = assignments_path
        self.distances_path = distances_path
        self.lock = lock
        self.log('\nset up on rank={rank}\n'.format(rank=self.rank))
    
    def assign_triangle(self, traj_index):
        self.log('assigning %s' % traj_index)
        traj = self.project.LoadTraj(traj_index)
        ptraj = self.metric.prepare_trajectory(traj)
        traj_length = len(ptraj)
        self.log('traj length: %s' % traj_length)
        recent_assignments = RotatingCache(5)
        assignments = np.zeros(traj_length, 'int')
        distances = np.zeros(traj_length, 'float32')
        
        for i in xrange(traj_length):
            prev_distances = self.metric.one_to_many(ptraj, self.pmetagens, i, recent_assignments.contents)
            meta_distances = self.metric.one_to_all(ptraj, self.pmetagens, i)
            print 'meta_distances', meta_distances
            close_gens_i = self.tfilter(meta_distances, prev_distances)
            
            d = self.metric.one_to_many(ptraj, self.pgens, i, close_gens_i)
            
            print i, close_gens_i
            # get best assignment from close_gens_i
            if len(d) > 0:
                argmin = np.argmin(d)
                assignment = close_gens_i[argmin]
                distance = d[argmin]
            else:
                distance = np.inf
            
            # get best assignment from prev
            if len(recent_assignments) > 0:
                prev_argmin = np.argmin(prev_distances)
                prev_min_distance = prev_distances[prev_argmin]
                # if the prev one is better, use it
                if prev_min_distance < distance:
                    distances[i] = prev_min_distance
                    assignments[i] = recent_assignments[prev_argmin]
                else:
                    assignments[i] = assignment
                    distances[i] = distance
            
            if assignments[i] not in recent_assignments:
                recent_assignments.push(assignments[i])
        
        s = Serializer({'assignments': assignments, 'distances': distances,
                                  'traj_index': traj_index})
        checkpoint_filename = os.path.join(self.chkpt_dir, 'trj_%d.chk' % traj_index)
        s.SaveToHDF(checkpoint_filename)
    
    def assign_notriangle(self, traj_index):
        self.log('Assigning %s' % traj_index)
        
        traj = self.project.LoadTraj(traj_index)
        ptraj = self.metric.prepare_trajectory(traj)
        traj_length = len(ptraj)
        assignments = np.zeros(traj_length, 'int')
        distances = np.zeros(traj_length, 'float32')
        
        for i in xrange(traj_length):
            d = self.metric.one_to_all(ptraj, self.pgens, i)
            assignments[i] = np.argmin(d)
            distances[i] = d[assignments[i]]

        lock_object = open(self.lock, 'r+')
        portalocker.lock(lock_object, portalocker.LOCK_EX)
        all_assignments = Serializer.LoadFromHDF(self.assignments_path)
        all_assignments['Data'][traj_index] = assignments
        all_distances = Serializer.LoadFromHDF(self.distances_path)
        all_distances['Data'][traj_index] = distances
        all_assignments['completed_trajectories'][traj_index] = True
        all_assignments.SaveToHDF(self.assignments_tmp)
        all_distances.SaveToHDF(self.distances_tmp)
        os.rename(self.assignments_tmp, self.assignments_path)
        os.rename(self.distances_tmp, self.distances_path)
        lock_object.close()

        
class TriangleFilter(object):
    def __init__(self, metric, centers_i, traj):
        pdata = metric.prepare_trajectory(traj)
        pcenters = pdata[centers_i]
        
        n_centers = len(centers_i)
        n_data = len(pdata)
        
        #self.centers_to_data[i,j] gives the distance from the ith center to the jth data
        centers_to_data = -1 * np.ones((n_centers, n_data))
        for i in xrange(n_centers):
            centers_to_data[i] = metric.one_to_all(pcenters, pdata, i)
        
        # assigned_data_list[i] is a list of the data assigned to center i
        # we're going to turn this into a numpy array instead
        assigned_data_list = [[] for i in xrange(n_centers)]
        for j in xrange(n_data):
            i = np.argmin(centers_to_data[:,j])
            assigned_data_list[i].append(j)
        self.n_assigned = np.array([len(s) for s in assigned_data_list])
        
        self.assigned_data = np.zeros(n_centers, dtype='object')
        self.data_to_centers = np.zeros(n_centers, dtype='object')        
        for i in xrange(n_centers):
            self.assigned_data[i] = -1 * np.ones(self.n_assigned[i], dtype='int')
            self.data_to_centers[i] = -1 * np.ones(self.n_assigned[i], dtype='float32')
            for j in xrange(self.n_assigned[i]):
                self.assigned_data[i][j] = assigned_data_list[i][j]
                self.data_to_centers[i][j] = centers_to_data[i, self.assigned_data[i][j]]
        
        self.cluster_radii = np.array([np.max(self.data_to_centers[i]) for i in xrange(n_centers)])
        self.worst_cluster_radii = np.max(self.cluster_radii)
        del assigned_data_list
    
    def __call__(self, new_to_centers, new_to_others=None):
        '''Given a vector of distances from a new point to the centers,
        return a set of the indicies of the data points which could possibly
        be the closest data to the new point.
        
        If new_to_others is supplied, it should be a list of distances of the new
        point to a subset of the data (besides the centers) that you think it might be
        close to.
        '''
        
        # new_to_centers should be a list of length equal to the numbers of centers
        
        results = set()
        ordered_centers = np.argsort(new_to_centers)
        
        closest_center = ordered_centers[0]
        minmax = new_to_centers[closest_center] + self.cluster_radii[closest_center]
        if new_to_others is not None and len(new_to_others) > 0:
            minmax = min(minmax, np.min(new_to_others))
        
        # factor this vector computation out of loop
        nc_wd = new_to_centers - self.cluster_radii
        mmwcr = minmax + self.worst_cluster_radii
        for c in ordered_centers:
            if nc_wd[c] > minmax:
                continue
            min_ds = new_to_centers[c] - self.data_to_centers[c]
            results |= set(self.assigned_data[c][np.where(min_ds <= minmax)[0]])
            if new_to_centers[c] > mmwcr:
                break
        
        return np.array(list(results))
    


class RotatingCache(object):
    def __init__(self, length, dtype='int'):
        self._i = 0
        self._max_length = length
        self._current_length = 0
        self._ary = np.zeros(self._max_length, dtype=dtype)

    def push(self, x):
        self._ary[self._i] = x
        self._i  = (self._i + 1) % self._max_length
        if self._current_length < self._max_length:
            self._current_length += 1

    @property
    def contents(self):
        if self._current_length < self._max_length:
            return self._ary[0:self._current_length]
        return self._ary

    def __contains__(self, x):
        if self._current_length < self._max_length:
            return x in self._ary[0:self._current_length]
        return x in self._ary

    def __getitem__(self, i):
        return self._ary[i]

    def __len__(self):
        return self._current_length
