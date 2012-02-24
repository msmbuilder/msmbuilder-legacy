import sys, os
from collections import deque
import numpy as np

from Emsmbuilder.Serializer import Serializer
from Emsmbuilder.Trajectory import Trajectory
from Emsmbuilder.clustering import _clarans, empty_trajectory_like
from Emsmbuilder.utils import uneven_zip
try:
    from Emsmbuilder import mm
except:
    pass
from msmbuilder.DistanceMetric import RMSD

def simple_assign(metric, project, generators, assignments_path, distances_path):
    """
    Assign each of the frames in each of the trajectories in the supplied project to
    their closest generator (frames of the trajectory "generators") using the supplied
    distance metric.
    
    assignments_path and distances_path should be the filenames of to h5 files in which the
    results are/will be stored. The results are stored along the way as they are computed,
    and if this method is killed halfway through execution, you can restart it and it
    will not have lost its place (i.e. checkpointing)
    """
    
    #import hashlib
    #print "Generators xyzlist hash", hashlib.sha1(generators['XYZList']).hexdigest()
    
    
    pgens = metric.prepare_trajectory(generators)
    
    #print 'PGens XYZData hash', hashlib.sha1(pgens.XYZData).hexdigest()
    
    
    
    num_gens = len(pgens)
    num_trajs = project['NumTrajs']
    longest = max(project['TrajLengths'])
    
    # setup tmp files
    assignments_tmp = assignments_path + '.tmp'
    distances_tmp = distances_path + '.tmp'
    if os.path.exists(assignments_tmp):
        os.remove(assignments_tmp)
    if os.path.exists(distances_tmp):
        os.remove(distances_tmp)
    
    all_distances = -1 * np.ones((num_trajs, longest), dtype=np.float32)
    all_assignments = -1 * np.ones((num_trajs, longest), dtype=np.int)
    
    if os.path.exists(assignments_path) and os.path.exists(distances_path):
        print assignments_path
        s = Serializer.LoadFromHDF(assignments_path)
        t = Serializer.LoadFromHDF(distances_path)
        all_distances = s['Data']
        all_assignments = t['Data']
        
        try:
            completed_trajectories = s['completed_trajectories']
        except:
            completed_trajectories = (all_assignments[:,0] >= 0)
        
    else:
        print "Creating serializer containers"
        completed_trajectories = np.array([False] * num_trajs)
        Serializer({'Data': all_assignments,
                    'completed_trajectories': completed_trajectories
                    }).SaveToHDF(assignments_path)
        Serializer({'Data': all_distances}).SaveToHDF(distances_path)
    
    for i in xrange(project['NumTrajs']):
        if completed_trajectories[i]:
            print 'Skipping trajectory %d -- already assigned' % i
            continue
        print 'Assigning trajectory %d' % i
        
        ptraj = metric.prepare_trajectory(project.LoadTraj(i))
        distances = -1 * np.ones(len(ptraj), dtype=np.float32)
        assignments = -1 * np.ones(len(ptraj), dtype=np.int)
        
        #for j in xrange(num_gens):
        #    d = metric.one_to_all(pgens, ptraj, j)
        #    where = np.where(d < distances)[0]
        #    
        #    distances[where] = d
        #    assignments[where] = j
        
        p2traj = RMSD.PrepareData(project.LoadTraj(i)['XYZList'][:, metric.atomindices])
        p2gens = RMSD.PrepareData(generators['XYZList'][:, metric.atomindices])
        
        for j in range(len(ptraj)):
            d = metric.one_to_all(ptraj, pgens, j)
            ind = np.argmin(d)
            assignments[j] = ind
            distances[j] = d[ind]
            

            #d2 = RMSD.GetFastMultiDistance(p2traj, p2gens, j)
            #np.testing.assert_array_equal(d, d2)
        
            #if i == 1 and j == 35:
            #    print '1,35'
            #    print "dist %.10f" % distances[35]
            #    print 'assign ', assignments[35]
            #    print 'ind ', ind
            #    import hashlib
            #    print 'atomindices hash', hashlib.sha1(metric.atomindices).hexdigest()
            #    print 'ptraj ', hashlib.sha1(ptraj.XYZData).hexdigest()
            #    print 'pgens ', hashlib.sha1(pgens.XYZData).hexdigest()
            #    print 'p2gens', hashlib.sha1(p2gens.XYZData).hexdigest()
            #    print '\n d[38]=%.10f' % d[38]
            #    print '\nd2[38]=%.10f' % d2[38]
            #    return (0,0)
            
        all_distances[i, 0:len(ptraj)] = distances
        all_assignments[i, 0:len(ptraj)] = assignments
        completed_trajectories[i] = True
        Serializer({'Data': all_assignments,
                    'completed_trajectories': completed_trajectories
                    }).SaveToHDF(assignments_tmp)
        Serializer({'Data': all_distances}).SaveToHDF(distances_tmp)
        os.rename(assignments_tmp, assignments_path)
        os.rename(distances_tmp, distances_path)
    
    return all_assignments, all_distances

# ###################### ###################### #####################
#
# BELOW IS SOME EXPERIMENTAL STUFF TO DO ASSIGNING IN PARALLEL VIA MPI
#               It's probably not working right now.
## ###################### ###################### #####################


class MasterAssigner(object):
    def __init__(self, metric, gens, project, checkpoint_dir):
        self.metric = metric
        self.gens = gens
        self.pgens = self.metric.prepare_trajectory(gens)
        self.checkpoint_dir = checkpoint_dir
        self.project = project
        
        #if triangle_inq:
        self._setup_triange()
  
    def _setup_triange(self):
        metagens_k = 24
        metagens_i = _clarans(self.metric, self.pgens, metagens_k, num_local_minima=5, max_neighbors=5, local_swap=True, verbose=True)[0]
        self.tfilter = TriangleFilter(self.metric, metagens_i, self.gens)
        self.metagens = empty_trajectory_like(self.gens)
        self.metagens['XYZList'] = self.gens['XYZList'][metagens_i]
        
    def assign(self, simple=True):
        mm.initialize_workers((self.project, self.metric, self.gens, self.tfilter, self.metagens, self.checkpoint_dir))
        jobs = [(i,) for i in range(self.project['NumTrajs'])]
        if simple:
            mm.map('assign_simple', jobs)
        else:
            mm.map('assign', jobs)
        
        

class SlaveAssigner(object):
    def __init__(self, project, metric, gens, tfilter, metagens, checkpoint_dir):
        self.project = project
        self.metric = metric
        self.pgens = metric.prepare_trajectory(gens)
        self.pmetagens = metric.prepare_trajectory(metagens)
        self.tfilter = tfilter
        self.checkpoint_dir = checkpoint_dir
        
    def assign(self, traj_index):
        print 'assigning %s' % traj_index
        traj = self.project.LoadTraj(traj_index)
        ptraj = self.metric.prepare_trajectory(traj)
        traj_length = len(ptraj)
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
        checkpoint_filename = os.path.join(self.checkpoint_dir, 'trj_%d.chk' % traj_index)
        s.SaveToHDF(checkpoint_filename)
    
    def assign_simple(self, traj_index):
        print 'assigning %s' % traj_index
        traj = self.project.LoadTraj(traj_index)
        ptraj = self.metric.prepare_trajectory(traj)
        traj_length = len(ptraj)
        recent_assignments = RotatingCache(5)
        assignments = np.zeros(traj_length, 'int')
        distances = np.zeros(traj_length, 'float32')
        
        for i in xrange(traj_length):
            d = self.metric.one_to_all(ptraj, self.pgens, i)
            assignments[i] = np.argmin(d)
            distances[i] = d[assignments[i]]
        s = Serializer({'assignments': assignments, 'distances': distances,
                        'traj_index': traj_index})
        checkpoint_filename = os.path.join(self.checkpoint_dir, 'trj_%d.chk' % traj_index)
        s.SaveToHDF(checkpoint_filename)
        
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
