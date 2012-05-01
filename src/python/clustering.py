import abc
import sys
import copy
import types
import random
import numpy as np
try:
    import fastcluster
except:
    print "Cannot import fastcluster."
import scipy.cluster.hierarchy


from msmbuilder import metrics, drift
from msmbuilder.Trajectory import Trajectory
from msmbuilder.Serializer import Serializer
from msmbuilder.utils import uneven_zip

from multiprocessing import Pool
try:
    from deap import dtm # has parallel map() implementation via mpi
except:
    pass

#####################################################################
#                                                                   #
#                       Begin Helper Functions                      #
#                                                                   #
#####################################################################

def concatenate_trajectories(trajectories):
    """Concatenate a list of trajectories into a single long trajectory"""
    #traj_lengths = [len(traj) for traj in trajectories]
    num_atoms = np.array([traj['XYZList'].shape[1] for traj in trajectories])
    if not np.all(num_atoms == num_atoms[0]):
        raise Exception('Not all same # atoms')
    
    result = empty_trajectory_like(trajectories[0])
        
    #result['XYZList'] = np.empty((sum(traj_lengths), num_atoms, 3) dtype='float32')
    result['XYZList'] = np.vstack([traj['XYZList'] for traj in trajectories])
    return result


def unconcatenate_trajectory(trajectory, lengths):
    """Take a single trajectory that was created by concatenating seperate
    trajectories and unconcenatenate it, returning the original trajectories.
    
    Note that you have to supply the lengths of the original trajectories.
    
    sum(lengths) should be equal to the length of the trajectory
    
    returns a list of length equal to len(lengths)
    """
    xyzlist = trajectory.pop('XYZList')
    empty = empty_trajectory_like(trajectory)
    output = [copy.deepcopy(empty) for length in lengths]
    xyzlists = split(xyzlist, lengths)
    for i, xyz in enumerate(xyzlists):
        output[i]['XYZList'] = xyz
    return output


def split(longlist, lengths):
    """Split a long list into segments"""
    if not sum(lengths) == len(longlist):
        raise Exception('sum(lengths)=%s, len(longlist)=%s' % (sum(lengths), len(longlist)))
    func = lambda (length, cumlength): longlist[cumlength-length:cumlength]
    iterable = zip(lengths, np.cumsum(lengths))
    output = map(func, iterable)
    return output


def stochastic_subsample(trajectories, shrink_multiple):
    """Given a list of trajectories, return a single trajectory
    shrink_multiple times smaller than the total number of frames in
    trajectories taken by random sampling of frames from trajectories
    
    Note that this method will modify the trajectory objects that you pass in
    
    @CHECK is the note above actually true?
    """
    shrink_multiple = int(shrink_multiple)
    if shrink_multiple < 1:
        raise ValueError('Shrink multiple should be an integer greater than 1. You supplied %s' % shrink_multiple)
    elif shrink_multiple == 1:
        #if isinstance(trajectories, Trajectory):
        #    return trajectories
        #return concatenate_trajectories(trajectories)
        return trajectories
    
    if isinstance(trajectories, Trajectory):
        traj = trajectories
        length = len(traj['XYZList'])
        new_length = int(length / shrink_multiple)
        if new_length <= 0:
            return None
        
        indices = np.array(random.sample(range(length), new_length))
        new_xyzlist = traj['XYZList'][indices, :, :]
        
        new_traj = empty_trajectory_like(traj)
        new_traj['XYZList'] = new_xyzlist
        return new_traj
        
    else:
        # assume we have a list of trajectories
        
        # check that all trajectories have the same number of atoms
        num_atoms = np.array([traj['XYZList'].shape[1] for traj in trajectories])
        if not np.all(num_atoms == num_atoms[0]):
            raise Exception('Not all same # atoms')
            
        # shrink each trajectory
        subsampled = [stochastic_subsample(traj, shrink_multiple) for traj in trajectories]
        # filter out failures
        subsampled = filter(lambda a: a is not None, subsampled)
        
        return concatenate_trajectories(subsampled)


def deterministic_subsample(trajectories, stride, start=0):
    """Given a list of trajectories, return a single trajectory
    shrink_multiple times smaller than the total number of frames in
    trajectories by taking every "stride"th frame, starting from "start"
    
    Note that this method will modify the trajectory objects that you pass in
    """
    
    stride = int(stride)
    if stride < 1:
        raise ValueError('stride should be an integer greater than 1. You supplied %s' % stride)
    elif stride == 1:
        #if isinstance(trajectories, Trajectory):
        #    return trajectories
        #return concatenate_trajectories(trajectories)
        return trajectories
    
    if isinstance(trajectories, Trajectory):
       traj = trajectories
       length = len(traj['XYZList'])
       traj['XYZList'] = traj['XYZList'][start::stride]
       return traj
    else:
       # assume we have a list of trajectories
       
       # check that all trajectories have the same number of atoms
       num_atoms = np.array([traj['XYZList'].shape[1] for traj in trajectories])
       if not np.all(num_atoms == num_atoms[0]):
           raise Exception('Not all same # atoms')
            
       # shrink each trajectory
       strided = [deterministic_subsample(traj, stride, start) for traj in trajectories]
       return concatenate_trajectories(strided)
    


def empty_trajectory_like(traj):
    """Return a trajectory with the same residue ids and stuff as traj, but NO xyzlist"""
    xyzlist = traj.pop('XYZList')
    new_traj = copy.deepcopy(traj)
    new_traj['XYZList'] = None
    traj['XYZList'] = xyzlist
    return new_traj    


def p_norm(Data, p=2):
    """Returns the p-norm of a Numpy array containing XYZ coordinates."""
    if p =="max":
        return Data.max()
    else:
        p=float(p)
        n=float(Data.shape[0])
        return ((Data**p).sum()/n)**(1/p)


#####################################################################
#                                                                   #
#                       End Helper Functions                        #
#                       Begin Clustering Function                   #
#                                                                   #
#####################################################################

def _assign(metric, ptraj, generator_indices):
    """Assign the frames in ptraj to the centers with indices *generator_indices*
    
    Arguments:
    metric - An instance of AbstractDistanceMetric capable of handling the prepared
             trajectory ptraj
    ptraj - A prepared trajectory, returned by the action of the preceding metric
            on a msmbuilder trajectory
    generator_indices - indices (with respect to ptraj) of the frames to be considered
                        the cluster centers.
    """
    
    assignments = np.zeros(len(ptraj), dtype='int')
    distances = np.inf * np.ones(len(ptraj), dtype='float32')
    for m in generator_indices:
        d = metric.one_to_all(ptraj, ptraj, m)
        closer = np.where(d < distances)[0]
        distances[closer] = d[closer]
        assignments[closer] = m
    return assignments, distances

def _kcenters(metric, ptraj, k=None, distance_cutoff=None, seed=0, verbose=True):
    """Run kcenters clustering algorithm.
    
    Terminates either when *k* clusters have been identified, or when every data
    is clustered better than *distance_cutoff*.
    
    Arguments:
    metric - An instance of AbstractDistanceMetric capable of handling the prepared
             trajectory ptraj
    ptraj - A prepared trajectory, returned by the action of the preceding metric
            on a msmbuilder trajectory
    k - number of desired clusters, or None. (int or None)
    distance_cutoff - Stop identifying new clusters once the distance of every data
                      to its cluster center falls below this value. (float or None)
    seed - index of the frame to use as the first cluster center. (int)
   
    Returns:
    generator_indices, assignments and distances.
    
    Note that the assignments that are
    are numbered with respect to the position in ptraj of the generator, not the
    position in generator_indices. That is, assignments[10] = 1020 means that the
    10th simulation frame is assigned to the 1020th simulation frame, not to the 1020th
    generator.
    """
    
    
    if k is None and distance_cutoff is None:
        raise ValueError("I need some cutoff criterion! both k and distance_cutoff can't both be none")
    if k is None and distance_cutoff <= 0:
        raise ValueError("With k=None you need to supply a legit distance_cutoff")
    if distance_cutoff is None:
        # set it below anything that can ever be reached
        distance_cutoff = -1
    if k is None:
        # set k to be the highest 32bit integer
        k = sys.maxint
    
    distance_list = np.inf * np.ones(len(ptraj), dtype=np.float32)
    assignments = -1 * np.ones(len(ptraj), dtype=np.int32) 
    
    generator_indices = []
    for i in xrange(k):
        new_ind = seed if i == 0 else np.argmax(distance_list)
        if distance_list[new_ind] < distance_cutoff:
            break
        new_distance_list = metric.one_to_all(ptraj, ptraj, new_ind)
        updated_indices = np.where(new_distance_list < distance_list)[0]
        distance_list[updated_indices] = new_distance_list[updated_indices]
        assignments[updated_indices] = new_ind
        generator_indices.append(new_ind)
        
    if verbose:
        print 'KCenters found %d generators' % i+1 # CRS added +1 
    
    return np.array(generator_indices), assignments, distance_list


def _clarans(metric, ptraj, k, num_local_minima, max_neighbors, local_swap=True, initial_medoids='kcenters', initial_assignments=None, initial_distance=None, verbose=True):
    """Run the CLARANS clustering algorithm on the frames in (prepared) trajectory
    *ptraj* using the distance metric *metric*. 
    
    Reference: Ng, R.T, Jan, Jiawei, 'CLARANS: A Method For Clustering Objects For
    Spatial Data Mining', IEEE Trans. on Knowledge and Data Engineering, vol. 14
    no.5 pp. 1003-1016 Sep/Oct 2002
    http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1033770
    
    Arguments:
    metric - An instance of AbstractDistanceMetric capable of handling the prepared
             trajectory ptraj
    ptraj - A prepared trajectory, returned by the action of the preceding metric
            on a msmbuilder trajectory
    k - number of desired clusters. (int)
    num_local_minima - number of local minima in the set of all possible clusterings
                       to identify. Execution time will scale linearly with this
                       parameter. The best of these local minima will be returned.
                       (int)
    max_neighbors - number of rejected swaps in a row necessary to declare a proposed
                    clustering a local minima (int)
    local_swap - If true, proposed swaps will be between a medoid and a data point
                 currently assigned to that medoid. If false, the data point for
                 the proposed swap is selected randomly. (boolean)
    initial_medoids - If 'kcenters', run kcenters clustering first to get
                      the initial medoids, and then run the swaps to improve it.
                      If 'random', select the medoids at random. Otherwise,
                      initial_medoids should be a numpy array of the indices of the
                      medoids.
    initial_assignments - If None, initial_assignments will be computed based on
                          the initial_medoids. If you pass in your own initial_medoids,
                          you can also pass in initial_assignments to avoid recomputing
                          them.
    initial_distances - If None, initial_distances will be computed based on
                        the initial_medoids. If you pass in your own initial_medoids,
                        you can also pass in initial_distances to avoid recomputing
                        them.
    verbose - Print information about the swaps being attempted. (boolean)
    
    Returns:
    center indices, assignments and distances
    """
    num_frames = len(ptraj)
    
    if initial_medoids == 'kcenters':
        initial_medoids, initial_assignments, initial_distance = _kcenters(metric, ptraj, k)
    elif initial_medoids == 'random':
        initial_medoids = np.random.permutation(np.arange(num_frames))[0:k]
        initial_assignments, initial_distance = _assign(metric, ptraj, initial_medoids)
    else:
        if not isinstance(initial_medoids, np.ndarray):
            raise ValueError('Initial medoids should be a numpy array')
        if initial_assignments is None or initial_distance is None:
            initial_assignments, initial_distance = _assign(metric, ptraj, initial_medoids)
    
    if not len(initial_assignments) == num_frames:
        raise ValueError('Initial assignments is not the same length as ptraj')
    if not len(initial_distance) == num_frames:
        raise ValueError('Initial distance is not the same length as ptraj')
    if not k == len(initial_medoids):
        raise ValueError('Initial medoids not the same length as k')
    
    if verbose:
        printer = sys.stdout
    else:
        printer = open('/dev/null', 'w')
        
        
    initial_pmedoids = ptraj[initial_medoids]
    initial_cost = np.sum(initial_distance)
    min_cost = initial_cost
    
    # these iterations could be parallelized
    for i in xrange(num_local_minima):
        print >> printer, '%s of %s local minima' % (i, num_local_minima)
        
        # the cannonical clarans approach is to initialize the medoids that you
        # start from randomly, but instead we use the kcenters medoids.
        
        medoids = initial_medoids
        pmedoids = initial_pmedoids
        assignments = initial_assignments
        distance_to_current = initial_distance
        current_cost = initial_cost
        
        optimal_medoids = initial_medoids
        optimal_assignments = initial_assignments
        optimal_distances = initial_distance
        
        #loop over neighbors
        j = 0
        while j < max_neighbors:
            medoid_i = np.random.randint(k)
            old_medoid = medoids[medoid_i]
            
            if local_swap is False:
                trial_medoid = np.random.randint(num_frames)
            else:
                trial_medoid = random.choice(np.where(assignments == medoids[medoid_i])[0])
                        
            new_medoids = medoids.copy()
            new_medoids[medoid_i] = trial_medoid
            pmedoids = ptraj[new_medoids]
            if type(pmedoids) == np.ndarray:
                pmedoids = pmedoids.copy()
                            
            new_distances = distance_to_current.copy()
            new_assignments = assignments.copy()
            
            print >> printer, '  swapping %s for %s...' % (old_medoid, trial_medoid),
            
            distance_to_trial = metric.one_to_all(ptraj, ptraj, trial_medoid)
            assigned_to_trial = np.where(distance_to_trial < distance_to_current)[0]
            new_assignments[assigned_to_trial] = trial_medoid
            new_distances[assigned_to_trial] = distance_to_trial[assigned_to_trial]
            
            ambiguous = np.where((new_assignments == old_medoid) & \
                                 (distance_to_trial >= distance_to_current))[0]
            for l in ambiguous:
                if len(ptraj) <= l:
                    print len(ptraj)
                    print l
                    print ptraj.dtype
                    print l.dtype
                #print len(ptraj), l
                d = metric.one_to_all(ptraj, pmedoids, l)
                argmin =  np.argmin(d)
                new_assignments[l] = new_medoids[argmin]
                new_distances[l] = d[argmin]
            
            new_cost = np.sum(new_distances)
            if new_cost < current_cost:
                print >> printer, 'Accept'
                medoids = new_medoids
                assignments = new_assignments
                distance_to_current = new_distances
                current_cost = new_cost
                
                j = 0
            else:
                j += 1
                print >> printer, 'Reject'
        
        if current_cost < min_cost:
            min_cost = current_cost
            optimal_medoids = medoids.copy()
            optimal_assignments = assignments.copy()
            optimal_distances = distance_to_current.copy()
    
    
    return optimal_medoids, optimal_assignments, optimal_distances


def _clarans_helper(args):
    return _clarans(*args)


def _hybrid_kmedoids(metric, ptraj, k=None, distance_cutoff=None, num_iters=10, local_swap=True, norm_exponent=2.0, too_close_cutoff=0.0001, ignore_max_objective=False, initial_medoids='kcenters', initial_assignments=None, initial_distance=None):
    """Run the hybrid kmedoids clustering algorithm from the msmbuilder2 paper on
    *ptraj* using the distance metric *metric*.
    
    Arguments:
    metric - An instance of AbstractDistanceMetric capable of handling the prepared
             trajectory ptraj
    ptraj - A prepared trajectory, returned by the action of the preceding metric
            on a msmbuilder trajectory
    k - number of desired clusters. (int)
    num_iters - number of swaps to attempt per medoid. (int)
    local_swap - If true, proposed swaps will be between a medoid and a data point
                 currently assigned to that medoid. If false, the data point for
                 the proposed swap is selected randomly. (boolean)
    norm_exponent - exponent to use in pnorm of the distance to generate objective
                    function (float)
    too_close_cutoff - Summarily reject proposed swaps if the distance of the medoid
                       to the trial medoid is less than thus value (float)
    ignore_max_objective - Ignore changes to the distance of the worst classified point,
                           and only reject or accept swaps based on changes to the
                           p norm of all the data points. (boolean)
    initial_medoids - If 'kcenters', run kcenters clustering first to get
                     the initial medoids, and then run the swaps to improve it.
                     If 'random', select the medoids at random. Otherwise,
                     initial_medoids should be a numpy array of the indices of the
                     medoids.
    initial_assignments - If None, initial_assignments will be computed based on
                         the initial_medoids. If you pass in your own initial_medoids,
                         you can also pass in initial_assignments to avoid recomputing
                         them.
    initial_distances - If None, initial_distances will be computed based on
                       the initial_medoids. If you pass in your own initial_medoids,
                       you can also pass in initial_distances to avoid recomputing
                       them.
    """
    if k is None and distance_cutoff is None:
        raise ValueError("I need some cutoff criterion! both k and distance_cutoff can't both be none")
    if k is None and distance_cutoff <= 0:
        raise ValueError("With k=None you need to supply a legit distance_cutoff")
    if distance_cutoff is None:
        # set it below anything that can ever be reached
        distance_cutoff = -1
    
    num_frames = len(ptraj)
    if initial_medoids == 'kcenters':
        initial_medoids, initial_assignments, initial_distance = _kcenters(metric, ptraj, k, distance_cutoff)
    elif initial_medoids == 'random':
        if k is None:
            raise ValueError('You need to supply the number of clusters, k, you want')
        initial_medoids = np.random.permutation(np.arange(num_frames))[0:k]
        initial_assignments, initial_distance = _assign(metric, ptraj, initial_medoids)
    else:
        if not isinstance(initial_medoids, np.ndarray):
            raise ValueError('Initial medoids should be a numpy array')
        if initial_assignments is None or initial_distance is None:
            initial_assignments, initial_distance = _assign(metric, ptraj, initial_medoids)
    
    assignments = initial_assignments
    distance_to_current = initial_distance
    medoids = initial_medoids
    pgens = ptraj[medoids]
    k = len(initial_medoids)
    
    obj_func = p_norm(distance_to_current, p=norm_exponent)
    max_norm = p_norm(distance_to_current, p='max')
    
    for iteration in xrange(num_iters):
        for medoid_i in xrange(k):
            
            if local_swap is False:
                trial_medoid = np.random.randint(num_frames)
            else:
                trial_medoid = random.choice(np.where(assignments == medoids[medoid_i])[0])
            
            old_medoid = medoids[medoid_i]
            
            if old_medoid == trial_medoid:
                continue
            
            new_medoids = medoids.copy()
            new_medoids[medoid_i] = trial_medoid
            pmedoids = ptraj[new_medoids]
            
            new_distances = distance_to_current.copy()
            new_assignments = assignments.copy()
            
            print 'Sweep %d, swapping medoid %d (conf %d) for conf %d...' % (iteration, medoid_i, old_medoid, trial_medoid)
            
            distance_to_trial = metric.one_to_all(ptraj, ptraj, trial_medoid)
            if distance_to_trial[old_medoid] < too_close_cutoff:
                print 'Too close'
                continue
            
            assigned_to_trial = np.where(distance_to_trial < distance_to_current)[0]
            new_assignments[assigned_to_trial] = trial_medoid
            new_distances[assigned_to_trial] = distance_to_trial[assigned_to_trial]
            
            ambiguous = np.where((new_assignments == old_medoid) & \
                                 (distance_to_trial >= distance_to_current))[0]
            for l in ambiguous:
                d = metric.one_to_all(ptraj, pmedoids, l)
                argmin =  np.argmin(d)
                new_assignments[l] = new_medoids[argmin]
                new_distances[l] = d[argmin]
                        
            new_obj_func = p_norm(new_distances, p=norm_exponent)
            new_max_norm = p_norm(new_distances, p='max')
            print "New f = %f, Old f = %f     ||      New Max Norm %f,  Old Max Norm = %f" % (new_obj_func, obj_func, new_max_norm, max_norm)
            if new_obj_func < obj_func and (new_max_norm <= max_norm or ignore_max_objective is True):
                print "Accept"
                medoids = new_medoids
                assignments = new_assignments
                distance_to_current = new_distances
                obj_func = new_obj_func
                max_norm = new_max_norm
            else:
                print 'Reject'
    
    return medoids, assignments, distance_to_current


#####################################################################
#                                                                   #
#                       End Clustering Functions                    #
#                       Begin Clustering Classes                    #
#                                                                   #
#####################################################################

class Hierarchical(object):
    allowable_methods = ['single', 'complete', 'average', 'weighted',
                         'centroid', 'median', 'ward']
                         
    def __init__(self, metric, trajectories, method='single', precomputed_values=None):
        """Initialize a hierarchical clusterer using the supplied distance
        metric and method. Method should be one of the fastcluster linkage methods,
        namely 'single', 'complete', 'average', 'weighted', 'centroid', 'median',
        or 'ward'.
        
        trajectories can be *either* a single Trajectory object, or a list of Trajectory
        objects.
        
        precomputed_values is used internally to implement load_from_disk()
        """
        
        if precomputed_values is not None:
            precomputed_z_matrix, traj_lengths = precomputed_values
            if isinstance(precomputed_z_matrix, np.ndarray) and precomputed_z_matrix.shape[1] == 4:
                self.Z = precomputed_z_matrix
                self.traj_lengths = traj_lengths
                return
            else:
                raise Exception('Something is wrong')
        
        if not isinstance(metric, metrics.AbstractDistanceMetric):
            raise TypeError('must be abstract distance metrice')
        if not method in self.allowable_methods:
            raise ValueError("%s not in %s" % (method, str(self.allowable_methods)))
        if 'XYZList' in trajectories:
            trajectories = [trajectories]
        elif isinstance(trajectories, types.GeneratorType):
            trajectories = list(trajectories)

        
        self.traj_lengths = np.array([len(traj['XYZList']) for traj in trajectories])
        #self.ptrajs = [self.metric.prepare_trajectory(traj) for traj in self.trajectories]
        
        print 'Preparing...'
        flat_trajectory = concatenate_trajectories(trajectories)
        pflat_trajectory = metric.prepare_trajectory(flat_trajectory)
        
        print 'Getting all to all pairwise distance matrix...'
        dmat = metric.all_pairwise(pflat_trajectory)
        print 'Done'
        self.Z = fastcluster.linkage(dmat, method=method, preserve_input=False)
        print 'Got Z matrix'
        #self.Z = scipy.cluster.hierarchy.linkage(dmat, method=method)
    
    def _oneD_assignments(self, k=None, cutoff_distance=None):
        """Assign the frames into clusters. Either supply k, the number of clusters
        desired, or cutoff_distance, a max diameteric of each cluster
        
        Returns a 1D array with the assignments of the flattened trajectory (internal).
        """
        # note that we subtract 1 from the results given by fcluster since
        # they start with 1-based numbering, but we want the lowest index cluster
        # to be number 0
        
        if k is not None and cutoff_distance is not None:
            raise Exception('You cant supply both a k and cutoff distance')
        elif k is not None:
            return scipy.cluster.hierarchy.fcluster(self.Z, k, criterion='maxclust') - 1
        elif cutoff_distance is not None:
            return scipy.cluster.hierarchy.fcluster(self.Z, cutoff_distance, criterion='distance') - 1
        else:
            raise Exception('You need to supply either k or a cutoff distance')
    
    def get_assignments(self, k=None, cutoff_distance=None):
        """Assign the frames into clusters. Either supply k, the number of clusters
        desired, or cutoff_distance, a max diameteric of each cluster
        
        Returns a 2D array padded with -1s
        """
        assgn_list = split(self._oneD_assignments(k, cutoff_distance), self.traj_lengths)
        output = -1 * np.ones((len(self.traj_lengths), max(self.traj_lengths)), dtype='int')
        for i, traj_assign in enumerate(assgn_list):
            output[i][0:len(traj_assign)] = traj_assign
        return output
    
    def save_to_disk(self, filename):
        s = Serializer({'z_matrix': self.Z, 'traj_lengths': self.traj_lengths})
        s.SaveToHDF(filename)
    
    @classmethod
    def load_from_disk(cls, filename):
        s = Serializer.LoadFromHDF(filename)
        Z, traj_lengths = s['z_matrix'], s['traj_lengths']
        #Next two lines are a hack to fix Serializer bug. KAB
        if np.rank(traj_lengths)==0:
            traj_lengths = [traj_lengths]
        return cls(None, None, precomputed_values=(Z, traj_lengths))
    


class BaseFlatClusterer(object):
    """
    (Abstract) base class / mixin that Clusterers can extend. Provides convenience
    functions for the user.
    
    To implement a clusterer using this base class, subclass it and define your
    init method to do the clustering you want, and then set self._generator_indices,
    self._assignments, and self._distances with the result.
    
    For convenience (and to enable some of its functionality), let BaseFlatCluster
    prepare the trajectories for you by calling BaseFlatClusterer's __init__ method
    and then using the prepared, concatenated trajectory self.ptraj for your clustering.
    """
    
    def __init__(self, metric, trajectories):
        if not isinstance(metric, metrics.AbstractDistanceMetric):
            raise TypeError('must be abstract distance metrice')
        # if we got a single trajectory instead a list of trajectories, make it a
        # list
        if 'XYZList' in trajectories:
            trajectories = [trajectories]
        elif isinstance(trajectories, types.GeneratorType):
            trajectories = list(trajectories)
        
        self._metric = metric
        self._traj_lengths = [len(traj) for traj in trajectories]
        self._concatenated = concatenate_trajectories(trajectories)
        self.ptraj = metric.prepare_trajectory(self._concatenated)
        self.num_frames = sum(self._traj_lengths)
        
        # All the actual Clusterer objects that subclass this base class
        # need to calculate these three parameters and store them here
        # self._generator_indices[i] = j means that the jth frame of self.ptraj is 
        # considered a generator. self._assignments[i] = j should indicate that
        # self.ptraj[j] is the coordinates of the the cluster center corresponding to i
        # and self._distances[i] = f should indicate that the distance from self.ptraj[i]
        # to  self.ptraj[self._assignments[i]] is f.
        self._generator_indices = 'abstract'
        self._assignments = 'abstract'
        self._distances = 'abstract'
    
    def _ensure_generators_computed(self):
        if self._generator_indices == 'abstract':
            raise Exception('Your subclass of BaseFlatClusterer is implemented wrong and didnt compute self._generator_indicies.')
    
    def _ensure_assignments_and_distances_computed(self):
        if self._assignments == 'abstract' or self._distances == 'abstract':
            self._assignments, self._distances = _assign(self._metric, self.ptraj, self._generator_indices)    
    
    def get_assignments(self):
        """Assign the trajectories you passed into the constructor based on generators that have
        been identified
        
        Returns:
        2D array of assignments where k = assignments[i,j] means that the
        jth frame in the ith trajectory is assigned to the center whose coordinates are
        in the kth frame of the trajectory in get_generators_as_traj()
        """
        self._ensure_generators_computed()
        self._ensure_assignments_and_distances_computed()
        
        twoD = split(self._assignments, self._traj_lengths)
        
        # the numbers in self._assignments are indices with respect to self.ptraj,
        # but we want indices with respect to the number in the trajectory of generators
        # returned by get_generators_as_traj()
        ptraj_index_to_gens_traj_index = np.zeros(self.num_frames)
        for i, g in enumerate(self._generator_indices):
            ptraj_index_to_gens_traj_index[g] = i
        
        # put twoD into a rectangular array
        output = -1 * np.ones((len(self._traj_lengths), max(self._traj_lengths)), dtype=np.int32)
        for i, traj_assign in enumerate(twoD):
            output[i,0:len(traj_assign)] = ptraj_index_to_gens_traj_index[traj_assign]
        
        return output
    
    def get_distances(self):
        self._ensure_generators_computed()
        self._ensure_assignments_and_distances_computed()
        
        twoD = split(self._distances, self._traj_lengths)
        
        # put twoD into a rectangular array
        output = -1 * np.ones((len(self._traj_lengths), max(self._traj_lengths)), dtype='float32')
        for i, traj_distances in enumerate(twoD):
            output[i][0:len(traj_distances)] = traj_distances
        return output
    
    def assign_new_trajectories(self, trajectories, checkpoint_callback=None):
        """Assign some new trajectories based on the generators identified by from the trajectory
        you passed to the constructor
        
        if you supply a checkpoint_callback, it should be a function that takes
        two arguments: the first is the trajectory index and the second is the
        1D array of assignments. It will get called after each trajectory in trajectories
        is assigned. DO NOT MODIFY the assignments (the 2nd argument), or else you
        will break everything.
        """
        self._ensure_generators_computed()
        
        if checkpoint_callback is None:
            checkpoint_callback = lambda i, a: sys.stdout.write('Assigned %d, length %d\n' % (i, len(a)))
        
        lengths = [len(traj['XYZList']) for traj in trajectories]
        assignments = -1 * np.ones(len(lengths), max(lengths), dtype='int')
        new_ptrajs = [self._metric.prepare_trajectory(traj) for traj in trajectories]
        
        pgens = self.ptraj[self._generator_indices]
        
        for i, new_traj in enumerate(trajectories):
            new_ptraj = self._metric.prepare_trajectory(new_traj)
            for j in xrange(len(new_traj['XYZList'])):
                d = self._metric.one_to_all(new_ptraj, pgens, j)
            assignments[i,j] = np.argmin(d)
            checkpoint_callback(i, assignments[i, :])
        
        return assignments
    
    
    def get_generators_as_traj(self):
        """Return a trajectory object where each frame is one of the generators/medoids identified"""
        self._ensure_generators_computed()
        
        output = empty_trajectory_like(self._concatenated)
        output['XYZList'] = self._concatenated['XYZList'][self._generator_indices, :, :]
        return output
    
    def save_to_disk(self, filename):
        # save generator_indices, metric, 
        raise NotImplementedError('sorry')
    
    @classmethod
    def load_from_disk(cls, filename):
        raise NotImplementedError('sorry')
    


class KCenters(BaseFlatClusterer):
    def __init__(self, metric, trajectories, k=None, distance_cutoff=None, seed=0):
        """'Run kcenters clustering algorithm.
        
        Terminates either when *k* clusters have been identified, or when every data
        is clustered better than *distance_cutoff*.
        
        Arguments:
        metric - An instance of AbstractDistanceMetric capable of handling the prepared
                 trajectory ptraj
        trajectories - A single trajectory, or an interable of trajectories that you
                       want to cluster.
        k - number of desired clusters, or None. (int or None)
        distance_cutoff - Stop identifying new clusters once the distance of every data
                          to its cluster center falls below this value. (float or None)
        seed - index of the frame to use as the first cluster center. (int)
        """
        super(KCenters, self).__init__(metric, trajectories)
        
        gi, asgn, dl = _kcenters(metric, self.ptraj, k, distance_cutoff, seed)
        
        # note that the assignments here are with respect to the numbering
        # in the trajectory -- they are not contiguous. Using the get_assignments()
        # method defined on the superclass (BaseFlatClusterer) will convert them
        # back into the contiguous numbering scheme (with respect to position in the
        # self._generator_indices).
        self._generator_indices = gi
        self._assignments = asgn
        self._distances = dl


class Clarans(BaseFlatClusterer):
    def __init__(self, metric, trajectories, k, num_local_minima=10, max_neighbors=20, local_swap=False):
        """Run the CLARANS clustering algorithm on the frames in (prepared) trajectory
        *ptraj* using the distance metric *metric*. 
        
        Reference: Ng, R.T, Jan, Jiawei, 'CLARANS: A Method For Clustering Objects For
        Spatial Data Mining', IEEE Trans. on Knowledge and Data Engineering, vol. 14
        no.5 pp. 1003-1016 Sep/Oct 2002
        http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1033770
        
        Arguments:
        metric - An instance of AbstractDistanceMetric capable of handling the prepared
                 trajectory ptraj
        trajectories - A single trajectory, or an interable of trajectories that you
                       want to cluster.
        k - number of desired clusters. (int)
        num_local_minima - number of local minima in the set of all possible clusterings
                           to identify. Execution time will scale linearly with this
                           parameter. The best of these local minima will be returned.
                           (int)
        max_neighbors - number of rejected swaps in a row necessary to declare a proposed
                        clustering a local minima (int)
        local_swap - If true, proposed swaps will be between a medoid and a data point
                     currently assigned to that medoid. If false, the data point for
                     the proposed swap is selected randomly. (boolean)
        """
        
        super(Clarans, self).__init__(metric, trajectories)
        
        medoids, assignments, distances = _clarans(metric, self.ptraj, k, num_local_minima, max_neighbors, local_swap, initial_medoids='kcenters')
                
        self._generator_indices = medoids
        self._assignments = assignments
        self._distances = distances


class SubsampledClarans(BaseFlatClusterer):
    def __init__(self, metric, trajectories, k, num_samples, shrink_multiple, num_local_minima=10,
                 max_neighbors=20, local_swap=False, parallel=None):
        """ Run the CLARANS algorithm (see the Clarans class for more description) on
        multiple subsamples of the data drawn randomly.
        
        Arguments:
        metric - An instance of AbstractDistanceMetric capable of handling the prepared
                 trajectory ptraj
        trajectories - A single trajectory, or an interable of trajectories that you
                       want to cluster.
        k - number of desired clusters. (int)
        
        num_samples - number of subsamples to draw (int)
        shrink_multiple - Each of the subsamples drawn will be of size equal to
                          the total number of frames divided by this number
        num_local_minima - number of local minima in the set of all possible clusterings
                           to identify. Execution time will scale linearly with this
                           parameter. The best of these local minima will be returned.
                           (int)
        max_neighbors - number of rejected swaps in a row necessary to declare a proposed
                        clustering a local minima (int)
        local_swap - If true, proposed swaps will be between a medoid and a data point
                     currently assigned to that medoid. If false, the data point for
                     the proposed swap is selected randomly. (boolean)
        parallel - Which parallelization library to use. ('multiprocessing', 'dtm', or None)
        """
        
        super(SubsampledClarans, self).__init__(metric, trajectories)
        
        if parallel is None:
            mymap = map
        elif parallel == 'multiprocessing':
            mymap = Pool().map
        elif parallel == 'dtm':
            mymap = dtm.map
        else:
            raise ValueError('Unrecognized parallelization')
        
        
        # function that returns a list of random indices
        gen_sub_indices = lambda: np.array(random.sample(range(self.num_frames), self.num_frames/shrink_multiple))
        #gen_sub_indices = lambda: np.arange(self.num_frames)
        
        sub_indices = [gen_sub_indices() for i in range(num_samples)]
        ptrajs = [self.ptraj[sub_indices[i]] for i in range(num_samples)]
        
        clarans_args = uneven_zip(metric, ptrajs, k, num_local_minima, max_neighbors, local_swap, ['kcenters'], None, None, False)
        
        results =  mymap(_clarans_helper, clarans_args)
        medoids_list, assignments_list, distances_list = zip(*results)
        best_i = np.argmin([np.sum(d) for d in distances_list])        
        
        #print 'best i', best_i
        #print 'best medoids (relative to subindices)', medoids_list[best_i]
        #print 'sub indices', sub_indices[best_i]
        #print 'best_medoids', sub_indices[best_i][medoids_list[best_i]]
        self._generator_indices = sub_indices[best_i][medoids_list[best_i]]
        


class HybridKMedoids(BaseFlatClusterer):
    def __init__(self, metric, trajectories, k, distance_cutoff=None, local_num_iters=10,
                       global_num_iters=0, norm_exponent=2.0, too_close_cutoff=.0001, ignore_max_objective=False):
        """Run the hybrid kmedoids clustering algorithm from the msmbuilder2 paper on
        *ptraj* using the distance metric *metric*.
        
        Arguments:
        metric - An instance of AbstractDistanceMetric capable of handling the prepared
                 trajectory ptraj
        trajectories - A single trajectory, or an interable of trajectories that you
                       want to cluster.
        k - number of desired clusters. (int)
        num_iters - number of swaps to attempt per medoid. (int)
        local_swap - If true, proposed swaps will be between a medoid and a data point
                     currently assigned to that medoid. If false, the data point for
                     the proposed swap is selected randomly. (boolean)
        norm_exponent - exponent to use in pnorm of the distance to generate objective
                        function (float)
        too_close_cutoff - Summarily reject proposed swaps if the distance of the medoid
                           to the trial medoid is less than thus value (float)
        ignore_max_objective - Ignore changes to the distance of the worst classified point,
                               and only reject or accept swaps based on changes to the
                               p norm of all the data points. (boolean)"""
        
        super(HybridKMedoids, self).__init__(metric, trajectories)
        
        
        medoids, assignments, distances = _hybrid_kmedoids(metric, self.ptraj, k, distance_cutoff,
                                                           local_num_iters, True, norm_exponent,
                                                           too_close_cutoff, ignore_max_objective,
                                                           initial_medoids='kcenters')
        if global_num_iters != 0:
            medoids, assignments, distances = _hybrid_kmedoids(metric, self.ptraj, k, distance_cutoff,
                                                           global_num_iters, False, norm_exponent,
                                                           too_close_cutoff, ignore_max_objective,
                                                           medoids, assignments, distances)
        
        self._generator_indices = medoids
        self._assignments = assignments
        self._distances = distances
        
#class KMeans(object):
#    def __init__(self, metric, trajectories, k, num_iters=1):
#        if not isinstance(metric, metrics.Vectorized):
#            raise TypeError('KMeans can only be used with Vectorized metrics')
#        if not metric.metric in ['euclidean', 'cityblock']:
#            raise TypeError('KMeans can only be used with euclidean or cityblock.')
#
#        self._traj_lengths = [len(traj['XYZList']) for traj in trajectories]
#        self._concatenated = concatenate_trajectories(trajectories)
#        self.ptraj = metric.prepare_trajectory(self._concatenated)
#
#        if metric.metric == 'euclidean':
#            d = 'e'
#        elif metric.metric == 'cityblock':
#            d = 'b'
#        else:
#            raise Exception('!')
#        
#        # seed with kcenters
#        indices, assignments, distances, = _kcenters(metric, self.ptraj, k=k, verbose=False)
#        ptraj_index_to_gens_traj_index = np.zeros(len(self.ptraj))
#        for i, g in enumerate(indices):
#            ptraj_index_to_gens_traj_index[g] = i
#        assignments = ptraj_index_to_gens_traj_index[assignments]
#
#        # now run kmeans
#        import Pycluster
#        assignments, error, nfound = Pycluster.kcluster(self.ptraj, nclusters=k, npass=num_iters, dist=d, initialid=assignments)
#
#        self._assignments = assignments
#
#        
#    def get_assignments(self):
#        assgn_list = split(self._assignments, self._traj_lengths)
#        output = -1 * np.ones((len(self._traj_lengths), max(self._traj_lengths)), dtype='int')
#        for i, traj_assign in enumerate(assgn_list):
#            output[i][0:len(traj_assign)] = traj_assign
#        return output
