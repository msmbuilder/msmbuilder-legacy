import sys
import types
import random
import numpy as np
try:
    import fastcluster
except ImportError:
    pass
import scipy.cluster.hierarchy

import mdtraj as md

from msmbuilder import metrics
from mdtraj import io
from msmbuilder.utils import uneven_zip

from multiprocessing import Pool
try:
    from deap import dtm  # has parallel map() implementation via mpi
except:
    pass

import logging
logger = logging.getLogger(__name__)

#####################################################################
#                                                                   #
#                       Begin Helper Functions                      #
#                                                                   #
#####################################################################


def concatenate_trajectories(trajectories):
    """Concatenate a list of trajectories into a single long trajectory

    Parameters
    ----------
    trajectories : list
        list of mdtraj.Trajectory object

    Returns
    -------
    concat_traj : mdtraj.Trajectory

    """

    assert len(trajectories) > 0, 'Please supply a list of trajectories'

    concat_traj = trajectories[0]
    for i in xrange(1, len(trajectories)):
        # Use mdtraj operator overloading
        concat_traj += trajectories[i]

    return concat_traj


def concatenate_prep_trajectories(prep_trajectories, metric):
    """Concatenate a list of prepared trajectories and
    create a single prepared_trajectory.

    This is non-trivial because the RMSD and LPRMSD prepared
    trajectories are not np.ndarrays ...

    Parameters
    ----------
    prep_trajectories : list
        list of prepared trajectories
    metric : msmbuilder.metrics.AbstractDistance metric subclass instance
        metric used to prepare the trajectories. Needed for RMSD and LPRMSD
        since concatenation requires recreating the prepared trajectory

    Returns
    -------
    ptraj : prepared_trajectory
        prepared trajectory instance, like that returned from
        metric.prepare_trajectory
    """
    if isinstance(prep_trajectories[0], np.ndarray):
        ptraj = np.concatenate(prep_trajectories)

    elif isinstance(prep_trajectories[0], RMSD.TheoData):

        xyz = np.concatenate([p.XYZData[:, :, :p.NumAtoms] for p in prep_trajectories])
        xyz = xyz.transpose((0, 2, 1))

        ptraj = metric.TheoData(xyz)

    else:
        raise Exception("unrecognized prepared trajectory."
            "NOTE: LPRMSD currently unsupported. Email schwancr@stanford.edu")

    return ptraj


def unconcatenate_trajectory(trajectory, lengths):
    """Take a single trajectory that was created by concatenating seperate
    trajectories and unconcenatenate it, returning the original trajectories.

    You have to supply the lengths of the original trajectories.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        Long trajectory to be split
    lengths : array_like
        list of lengths to split the long trajectory into

    Returns
    -------
    A list of trajectories
    """

    return split(trajectory, lengths)


def split(longlist, lengths):
    """Split a long list into segments

    Parameters
    ----------
    longlist : array_like
        Long trajectory to be split
    lengths : array_like
        list of lengths to split the long list into

    Returns
    -------
    A list of lists
    """

    if not sum(lengths) == len(longlist):
        raise Exception('sum(lengths)=%s, len(longlist)=%s' % (sum(lengths), len(longlist)))
    func = lambda (length, cumlength): longlist[cumlength - length: cumlength]
    iterable = zip(lengths, np.cumsum(lengths))
    output = map(func, iterable)
    return output


def stochastic_subsample(trajectories, shrink_multiple):
    """Randomly subsample from a trajectory

    Given a list of trajectories, return a single trajectory
    shrink_multiple times smaller than the total number of frames in
    trajectories taken by random sampling of frames from trajectories

    Parameters
    ----------
    trajectories : list of mdtraj.Trajectory
        list of trajectories to sample from
    shrink_multiple : int
        fraction to shrint by

    Note that this method will modify the trajectory objects that you pass in

    @CHECK is the note above actually true?
    """
    shrink_multiple = int(shrink_multiple)
    if shrink_multiple < 1:
        raise ValueError('Shrink multiple should be an integer greater than 1. You supplied %s' % shrink_multiple)
    elif shrink_multiple == 1:
        # if isinstance(trajectories, Trajectory):
        #    return trajectories
        # return concatenate_trajectories(trajectories)
        return trajectories

    if isinstance(trajectories, md.Trajectory):
        traj = trajectories
        length = traj.n_frames
        new_length = int(length / shrink_multiple)
        if new_length <= 0:
            return None

        indices = np.array(random.sample(range(length), new_length))
        new_traj = traj[indices, :, :]

        return new_traj

    else:
        # assume we have a list of trajectories

        # check that all trajectories have the same number of atoms
        num_atoms = np.array([traj.n_atoms for traj in trajectories])
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

    Parameters
    ----------
    trajectories : list of mdtraj.Trajectory
        trajectories to subsample from
    stride : int
        freq to subsample at
    start : int
        first frame to pick

    Returns
    -------
    trajectory : mdtraj.trajectory
        shortened trajectory
    """

    stride = int(stride)
    if stride < 1:
        raise ValueError('stride should be an integer greater than 1. You supplied %s' % stride)
    elif stride == 1:
        # if isinstance(trajectories, Trajectory):
        #    return trajectories
        # return concatenate_trajectories(trajectories)
        return trajectories

    if isinstance(trajectories, Trajectory):
        traj = trajectories
        traj = traj[start::stride]
        return traj
    else:
        # assume we have a list of trajectories

        # check that all trajectories have the same number of atoms
        num_atoms = np.array([traj.n_atoms for traj in trajectories])
        if not np.all(num_atoms == num_atoms[0]):
            raise Exception('Not all same # atoms')

        # shrink each trajectory
        strided = [deterministic_subsample(traj, stride, start) for traj in trajectories]
        return concatenate_trajectories(strided)


def p_norm(data, p=2):
    """p_norm of an ndarray with XYZ coordinates

    Parameters
    ----------
    data : ndarray
        XYZ coordinates. TODO: Shape?
    p : {int, "max"}, optional
        power of p_norm

    Returns
    -------
    value : float
        the answer
    """

    if p == "max":
        return data.max()
    else:
        p = float(p)
        n = float(data.shape[0])
        return ((data ** p).sum() / n) ** (1.0 / p)


#####################################################################
#                                                                   #
#                       End Helper Functions                        #
#                       Begin Clustering Function                   #
#                                                                   #
#####################################################################


def _assign(metric, ptraj, generator_indices):
    """Assign the frames in ptraj to the centers with indices *generator_indices*

    Parameters
    ----------
    metric : msmbuilder.metrics.AbstractDistanceMetric
        A metric capable of handling `ptraj`
    ptraj : prepared trajectory
        ptraj return by the action of the preceding metric on a msmbuilder trajectory
    generator_indices : array_like
        indices (with respect to ptraj) of the frames to be considered the
        cluster centers.

    Returns
    -------
    assignments : ndarray
        `assignments[i] = j` means that the `i`th frame in ptraj is assigned to
        `ptraj[j]`
    distances :  ndarray
        `distances[i] = j` means that the distance (according to `metric`) from
        `ptraj[i]` to `ptraj[assignments[i]]` is `j`
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

    Terminates either when `k` clusters have been identified, or when every data
    is clustered better than `distance_cutoff`.

    Parameters
    ----------
    metric : msmbuilder.metrics.AbstractDistanceMetric
        A metric capable of handling `ptraj`
    ptraj : prepared trajectory
        ptraj return by the action of the preceding metric on a msmbuilder trajectory
    k : {int, None}
        number of desired clusters, or None
    distance_cutoff : {float, None}
        Stop identifying new clusters once the distance of every data to its
        cluster center falls below this value. Supply either this or `k`
    seed : int, optional
        index of the frame to use as the first cluster center
    verbose : bool, optional
        print as each new generator is found

    Returns
    -------
    generator_indices : ndarray
        indices (with respect to ptraj) of the frames to be considered cluster centers
    assignments : ndarray
        the cluster center to which each frame is assigned to (1D)
    distances : ndarray
        distance from each of the frames to the cluster center it was assigned to

    See Also
    --------
    KCenters : wrapper around this implementation that provides more convenience

    Notes
    ------
    the assignments are numbered with respect to the position in ptraj of the
    generator, not the position in generator_indices. That is, assignments[10] =
    1020 means that the 10th simulation frame is assigned to the 1020th
    simulation frame, not to the 1020th generator.

    References
    ----------
    .. [1] Beauchamp, MSMBuilder2
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
        if k == sys.maxint:
            logger.info("K-centers: Finding generator %i. Will finish when % .4f drops below % .4f", i, distance_list[new_ind], distance_cutoff)
        else:
            logger.info("K-centers: Finding generator %i", i)

        if distance_list[new_ind] < distance_cutoff:
            break
        new_distance_list = metric.one_to_all(ptraj, ptraj, new_ind)
        updated_indices = np.where(new_distance_list < distance_list)[0]
        distance_list[updated_indices] = new_distance_list[updated_indices]
        assignments[updated_indices] = new_ind
        generator_indices.append(new_ind)

    if verbose:
        logger.info('KCenters found %d generators', i + 1)

    return np.array(generator_indices), assignments, distance_list


def _clarans(metric, ptraj, k, num_local_minima, max_neighbors, local_swap=True, initial_medoids='kcenters', initial_assignments=None, initial_distance=None, verbose=True):
    """Run the CLARANS clustering algorithm on the frames in a trajectory


    Reference
    ---------
    .. [1] Ng, R.T, Jan, Jiawei, 'CLARANS: A Method For Clustering Objects For
    Spatial Data Mining', IEEE Trans. on Knowledge and Data Engineering, vol. 14
    no.5 pp. 1003-1016 Sep/Oct 2002
    http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1033770

    Parameters
    ----------
    metric : msmbuilder.metrics.AbstractDistanceMetric
        A metric capable of handling `ptraj`
    ptraj : prepared trajectory
        ptraj return by the action of the preceding metric on a msmbuilder trajectory
    k : int
        number of desired clusters
    num_local_minima : int
        number of local minima in the set of all possible clusterings to identify.
        Execution time will scale linearly with this parameter. The best of
        these local minima will be returned.
    max_neighbors : int
        number of rejected swaps in a row necessary to declare a proposed
        clustering a local minima
    local_swap : bool, optional
        If true, proposed swaps will be between a medoid and a data point
        currently assigned to that medoid. If false, the data point for
        the proposed swap is selected randomly.
    initial_medoids : {'kcenters', 'random', ndarray}, optional
        If 'kcenters', run kcenters clustering first to get the initial medoids,
        and then run the swaps to improve it. If 'random', select the medoids at
        random. Otherwise, initial_medoids should be a numpy array of the
        indices of the medoids.
    initial_assignments : {None, ndarray}, optional
        If None, initial_assignments will be computed based on the
        initial_medoids. If you pass in your own initial_medoids, you can also
        pass in initial_assignments to avoid recomputing them.
    initial_distances : {None, ndarray}, optional
        If None, initial_distances will be computed based on the initial_medoids.
        If you pass in your own initial_medoids, you can also pass in
        initial_distances to avoid recomputing them.
    verbose : bool, optional
        Print information about the swaps being attempted

    Returns
    -------
    generator_indices : ndarray
        indices (with respect to ptraj) of the frames to be considered cluster centers
    assignments : ndarray
        the cluster center to which each frame is assigned to (1D)
    distances : ndarray
        distance from each of the frames to the cluster center it was assigned to
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


    initial_pmedoids = ptraj[initial_medoids]
    initial_cost = np.sum(initial_distance)
    min_cost = initial_cost

    # these iterations could be parallelized
    for i in xrange(num_local_minima):
        logger.info('%s of %s local minima', i, num_local_minima)

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

        # loop over neighbors
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

            logger.info('swapping %s for %s...', old_medoid, trial_medoid)

            distance_to_trial = metric.one_to_all(ptraj, ptraj, trial_medoid)
            assigned_to_trial = np.where(distance_to_trial < distance_to_current)[0]
            new_assignments[assigned_to_trial] = trial_medoid
            new_distances[assigned_to_trial] = distance_to_trial[assigned_to_trial]

            ambiguous = np.where((new_assignments == old_medoid) & \
                                 (distance_to_trial >= distance_to_current))[0]
            for l in ambiguous:
                if len(ptraj) <= l:
                    logger.error(len(ptraj))
                    logger.error(l)
                    logger.error(ptraj.dtype)
                    logger.error(l.dtype)

                d = metric.one_to_all(ptraj, pmedoids, l)
                argmin = np.argmin(d)
                new_assignments[l] = new_medoids[argmin]
                new_distances[l] = d[argmin]

            new_cost = np.sum(new_distances)
            if new_cost < current_cost:
                logger.info('Accept')
                medoids = new_medoids
                assignments = new_assignments
                distance_to_current = new_distances
                current_cost = new_cost

                j = 0
            else:
                j += 1
                logger.info('Reject')

        if current_cost < min_cost:
            min_cost = current_cost
            optimal_medoids = medoids.copy()
            optimal_assignments = assignments.copy()
            optimal_distances = distance_to_current.copy()


    return optimal_medoids, optimal_assignments, optimal_distances


def _clarans_helper(args):
    return _clarans(*args)


def _hybrid_kmedoids(metric, ptraj, k=None, distance_cutoff=None, num_iters=10, local_swap=True, norm_exponent=2.0, too_close_cutoff=0.0001, ignore_max_objective=False, initial_medoids='kcenters', initial_assignments=None, initial_distance=None):
    """Run the hybrid kmedoids clustering algorithm to cluster a trajectory

    References
    ----------
    .. [1] Beauchamp, K. MSMBuilder2

    Parameters
    ----------
    metric : msmbuilder.metrics.AbstractDistanceMetric
        A metric capable of handling `ptraj`
    ptraj : prepared trajectory
        ptraj return by the action of the preceding metric on a msmbuilder trajectory
    k : int
        number of desired clusters
    num_iters : int
        number of swaps to attempt per medoid
    local_swap : boolean, optional
        If true, proposed swaps will be between a medoid and a data point
        currently assigned to that medoid. If false, the data point for the
        proposed swap is selected randomly.
    norm_exponent : float, optional
        exponent to use in pnorm of the distance to generate objective function
    too_close_cutoff : float, optional
        Summarily reject proposed swaps if the distance of the medoid to the trial
        medoid is less than thus value
    ignore_max_objective : boolean, optional
        Ignore changes to the distance of the worst classified point, and only
        reject or accept swaps based on changes to the p norm of all the data
        points.
    initial_medoids : {'kcenters', ndarray}
        If 'kcenters', run kcenters clustering first to get the initial medoids,
        and then run the swaps to improve it. If 'random', select the medoids at
        random. Otherwise, initial_medoids should be a numpy array of the
        indices of the medoids.
    initial_assignments : {None, ndarray}, optional
        If None, initial_assignments will be computed based on the
        initial_medoids. If you pass in your own initial_medoids, you can also
        pass in initial_assignments to avoid recomputing them.
    initial_distances : {None, ndarray}, optional
        If None, initial_distances will be computed based on the initial_medoids.
        If you pass in your own initial_medoids, you can also pass in
        initial_distances to avoid recomputing them.

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

    if not np.all(np.unique(medoids) == np.sort(medoids)):
        raise ValueError('Initial medoids must be distinct')
    if not np.all(np.unique(assignments) == sorted(medoids)):
        raise ValueError('Initial assignments dont match initial medoids')

    for iteration in xrange(num_iters):
        for medoid_i in xrange(k):
            if not np.all(np.unique(assignments) == sorted(medoids)):
                raise ValueError('Loop invariant lost')

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

            logger.info('Sweep %d, swapping medoid %d (conf %d) for conf %d...', iteration, medoid_i, old_medoid, trial_medoid)

            distance_to_trial = metric.one_to_all(ptraj, ptraj, trial_medoid)
            if not np.all(np.isfinite(distance_to_trial)):
                raise ValueError('distance metric returned nonfinite distances')

            if distance_to_trial[old_medoid] < too_close_cutoff:
                logger.info('Too close')
                continue

            assigned_to_trial = np.where(distance_to_trial < distance_to_current)[0]
            new_assignments[assigned_to_trial] = trial_medoid
            new_distances[assigned_to_trial] = distance_to_trial[assigned_to_trial]

            ambiguous = np.where((new_assignments == old_medoid) & \
                                 (distance_to_trial >= distance_to_current))[0]
            for l in ambiguous:
                d = metric.one_to_all(ptraj, pmedoids, l)
                if not np.all(np.isfinite(d)):
                    raise ValueError('distance metric returned nonfinite distances')
                argmin = np.argmin(d)
                new_assignments[l] = new_medoids[argmin]
                new_distances[l] = d[argmin]

            new_obj_func = p_norm(new_distances, p=norm_exponent)
            new_max_norm = p_norm(new_distances, p='max')

            if new_obj_func < obj_func and (new_max_norm <= max_norm or ignore_max_objective is True):
                logger.info("Accept. New f = %f, Old f = %f", new_obj_func, obj_func)
                medoids = new_medoids
                assignments = new_assignments
                distance_to_current = new_distances
                obj_func = new_obj_func
                max_norm = new_max_norm
            else:
                logger.info("Reject. New f = %f, Old f = %f", new_obj_func, obj_func)

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
        metric and method.

        Method should be one of the fastcluster linkage methods,
        namely 'single', 'complete', 'average', 'weighted', 'centroid', 'median',
        or 'ward'.

        Parameters
        ----------
        metric : msmbuilder.metrics.AbstractDistanceMetric
            A metric capable of handling `ptraj`
        trajectory : Trajectory list of Trajectorys
            data to cluster
        method : {'single', 'complete', 'average', 'weighted', 'centroid',
                  'median', 'ward'}
        precomputed_values :
            used internally to implement load_from_disk()

        Notes
        -----
        This is implemenred with the fastcluster library, which can be downloaded
        from CRAN http://cran.r-project.org/web/packages/fastcluster/
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
            raise TypeError('%s is not an abstract distance metric' % metric)
        if not method in self.allowable_methods:
            raise ValueError("%s not in %s" % (method, str(self.allowable_methods)))
        if isinstance(trajectories, md.Trajectory):
            trajectories = [trajectories]
        elif isinstance(trajectories, types.GeneratorType):
            trajectories = list(trajectories)


        self.traj_lengths = np.array([len(t) for t in trajectories])
        # self.ptrajs = [self.metric.prepare_trajectory(traj) for traj in self.trajectories]

        logger.info('Preparing...')
        flat_trajectory = concatenate_trajectories(trajectories)
        pflat_trajectory = metric.prepare_trajectory(flat_trajectory)

        logger.info('Getting all to all pairwise distance matrix...')
        dmat = metric.all_pairwise(pflat_trajectory)
        logger.info('Done with all2all')
        self.Z = fastcluster.linkage(dmat, method=method, preserve_input=False)
        logger.info('Got Z matrix')
        # self.Z = scipy.cluster.hierarchy.linkage(dmat, method=method)

    def _oneD_assignments(self, k=None, cutoff_distance=None):
        """Assign the frames into clusters.

        Either supply k, the number of clusters desired, or cutoff_distance, a
        max diameteric of each cluster

        Parameters
        ----------
        k : int, optional
            number of clusters desired
        cutoff_distance : float, optional
            max diameter of each cluster, as a cutoff

        Returns
        -------
        assignments_1d : ndarray
            1D array with the assignments of the flattened trajectory (internal).

        See Also
        --------
        Hierarchical.get_assignments

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
        """Assign the frames into clusters.

        Either supply k, the number of clusters desired, or cutoff_distance, a
        max diameteric of each cluster

        Parameters
        ----------
        k : int, optional
            number of clusters desired
        cutoff_distance : float, optional
            max diameter of each cluster, as a cutoff

        Returns
        -------
        assignments : ndarray
            2D array of shape num_trajs x length of longest traj. Padded with -1s
            at the end if not all trajectories are the same length
        """
        assgn_list = split(self._oneD_assignments(k, cutoff_distance), self.traj_lengths)
        output = -1 * np.ones((len(self.traj_lengths), max(self.traj_lengths)), dtype='int')
        for i, traj_assign in enumerate(assgn_list):
            output[i][0:len(traj_assign)] = traj_assign
        return output

    def save_to_disk(self, filename):
        """Save this clusterer to disk.

        This is useful because computing the Z-matrix
        (done in __init__) is the most expensive part, and assigning is cheap

        Parameters
        ----------
        filename : str
            location to save to

        Raises
        ------
        Exception if something already exists at `filename`
        """
        io.saveh(filename, z_matrix=self.Z, traj_lengths=self.traj_lengths)

    @classmethod
    def load_from_disk(cls, filename):
        """Load up a clusterer from disk

        This is useful because computing the Z-matrix
        (done in __init__) is the most expensive part, and assigning is cheap

        Parameters
        ----------
        filename : str
            location to save to

        Raises
        ------
        TODO: Probablt raises something if filename doesn't exist?
        """
        data = io.loadh(filename, deferred=False)
        Z, traj_lengths = data['z_matrix'], data['traj_lengths']
        # Next two lines are a hack to fix Serializer bug. KAB
        if np.rank(traj_lengths) == 0:
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

    def __init__(self, metric, trajectories=None, prep_trajectories=None):

        if not isinstance(metric, metrics.AbstractDistanceMetric):
            raise TypeError('%s is not an AbstractDistanceMetric' % metric)
        # if we got a single trajectory instead a list of trajectories, make it a
        # list
        if not trajectories is None:
            if isinstance(trajectories, md.Trajectory):
                trajectories = [trajectories]
            elif isinstance(trajectories, types.GeneratorType):
                trajectories = list(trajectories)

            self._concatenated = concatenate_trajectories(trajectories)

            if prep_trajectories is None:
                self.ptraj = metric.prepare_trajectory(self._concatenated)

            self._traj_lengths = [len(traj) for traj in trajectories]

        if not prep_trajectories is None:  # If they also provide trajectories
            # that's fine, but we will use the prep_trajectories

            if isinstance(prep_trajectories, np.ndarray) or  \
                isinstance(prep_trajectories[0], np.ndarray):

                prep_trajectories = np.array(prep_trajectories)
                if len(prep_trajectories.shape) == 2:
                    prep_trajectories = [prep_trajectories]
                else:
                    # 3D means a list of prep_trajectories was input
                    prep_trajectories = list(prep_trajectories)

            if trajectories is None:
                self._traj_lengths = [len(ptraj) for ptraj in prep_trajectories]

            self._concatenated = None
            self.ptraj = concatenate_prep_trajectories(prep_trajectories, metric)

        if trajectories is None and prep_trajectories is None:
            raise Exception("must provide at least one of trajectories and prep_trajectories")

        self._metric = metric


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
        """Assign the trajectories you passed into the constructor based on
        generators that have been identified

        Returns
        -------
        assignments : ndarray
            2D array of assignments where k = assignments[i,j] means that the
            jth frame in the ith trajectory is assigned to the center whose
            coordinates are in the kth frame of the trajectory in
            get_generators_as_traj()
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
            output[i, 0:len(traj_assign)] = ptraj_index_to_gens_traj_index[traj_assign]

        return output

    def get_distances(self):
        """Extract the distance from each frame to its assigned cluster kcenter

        Returns
        -------
        distances : ndarray
            2D array of size num_trajs x length of longest traj, such that
            distances[i,j] gives the distance from the ith trajectorys jth
            frame to its assigned cluster center
        """
        self._ensure_generators_computed()
        self._ensure_assignments_and_distances_computed()

        twoD = split(self._distances, self._traj_lengths)

        # put twoD into a rectangular array
        output = -1 * np.ones((len(self._traj_lengths), max(self._traj_lengths)), dtype='float32')
        for i, traj_distances in enumerate(twoD):
            output[i][0:len(traj_distances)] = traj_distances
        return output

    def get_generators_as_traj(self):
        """Get a trajectory containing the generators

        Returns
        -------
        traj or ptraj : msmbuilder.Trajectory or np.ndarray
            a trajectory object where each frame is one of the
            generators/medoids identified. If trajectories was
            not originally provided, then will only return the
            prepared generators

        """
        self._ensure_generators_computed()

        if self._concatenated is None:
            output = self.ptraj[self._generator_indices]
        else:
            output = self._concatenated[self._generator_indices]

        return output

    def get_generator_indices(self):
        """Get the generator indices corresponding to frames in
        self.ptraj.

        Returns
        -------
        gen_inds : np.ndarray
            generator indices corresponding to the generators in
            self.ptraj
        """

        return self._generator_indices


class KCenters(BaseFlatClusterer):
    def __init__(self, metric, trajectories=None, prep_trajectories=None,
                 k=None, distance_cutoff=None, seed=0):
        """Run kcenters clustering algorithm.

        Terminates either when `k` clusters have been identified, or when every data
        is clustered better than `distance_cutoff`.

        Parameters
        ----------
        metric : msmbuilder.metrics.AbstractDistanceMetric
            A metric capable of handling `ptraj`
        trajectory : Trajectory or list of msmbuilder.Trajectory
            data to cluster
        k : {int, None}
            number of desired clusters, or None
        distance_cutoff : {float, None}
            Stop identifying new clusters once the distance of every data to its
            cluster center falls below this value. Supply either this or `k`
        seed : int, optional
            index of the frame to use as the first cluster center

        See Also
        --------
        _kcenters : implementation

        References
        ----------
        .. [1] Beauchamp, MSMBuilder2
        """

        super(KCenters, self).__init__(metric, trajectories, prep_trajectories)

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
    def __init__(self, metric, trajectories=None, prep_trajectories=None, k=None,
                 num_local_minima=10, max_neighbors=20, local_swap=False):
        """Run the CLARANS clustering algorithm on the frames in a trajectory

        Reference
        ---------
        .. [1] Ng, R.T, Jan, Jiawei, 'CLARANS: A Method For Clustering Objects For
        Spatial Data Mining', IEEE Trans. on Knowledge and Data Engineering, vol. 14
        no.5 pp. 1003-1016 Sep/Oct 2002
        http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1033770

        Parameters
        ----------
        metric : msmbuilder.metrics.AbstractDistanceMetric
            A metric capable of handling `ptraj`
        trajectory : Trajectory or list of msmbuilder.Trajectory
            data to cluster
        k : int
            number of desired clusters
        num_local_minima : int
            number of local minima in the set of all possible clusterings to identify.
            Execution time will scale linearly with this parameter. The best of
            these local minima will be returned.
        max_neighbors : int
            number of rejected swaps in a row necessary to declare a proposed
            clustering a local minima
        local_swap : bool, optional
            If true, proposed swaps will be between a medoid and a data point
            currently assigned to that medoid. If false, the data point for
            the proposed swap is selected randomly.

        See Also
        --------
        _kcenters : implementation
        SubsampledClarans : random subsampling version (faster)
        """

        super(Clarans, self).__init__(metric, trajectories, prep_trajectories)

        medoids, assignments, distances = _clarans(metric, self.ptraj, k,
            num_local_minima, max_neighbors, local_swap, initial_medoids='kcenters')

        self._generator_indices = medoids
        self._assignments = assignments
        self._distances = distances


class SubsampledClarans(BaseFlatClusterer):
    def __init__(self, metric, trajectories=None, prep_trajectories=None, k=None,
                 num_samples=None, shrink_multiple=None, num_local_minima=10,
                 max_neighbors=20, local_swap=False, parallel=None):
        """ Run the CLARANS algorithm (see the Clarans class for more description) on
        multiple subsamples of the data drawn randomly.

        Parameters
        ----------
        metric : msmbuilder.metrics.AbstractDistanceMetric
            A metric capable of handling `ptraj`
        trajectories : Trajectory or list of msmbuilder.Trajectory
            data to cluster
        prep_trajectories : np.ndarray or None
            prepared trajectories instead of msmbuilder.Trajectory
        k : int
            number of desired clusters
        num_samples : int
            number of random subsamples to draw
        shrink_multiple : int
            Each of the subsamples drawn will be of size equal to the total
            number of frames divided by this number
        num_local_minima : int, optional
            number of local minima in the set of all possible clusterings
            to identify. Execution time will scale linearly with this
            parameter. The best of these local minima will be returned.
        max_neighbors : int, optional
            number of rejected swaps in a row necessary to declare a proposed
            clustering a local minima
        local_swap : bool, optional
            If true, proposed swaps will be between a medoid and a data point
            currently assigned to that medoid. If false, the data point for
            the proposed swap is selected randomly
        parallel : {None, 'multiprocessing', 'dtm}
            Which parallelization library to use. Each of the random subsamples
            are run independently
        """

        super(SubsampledClarans, self).__init__(metric, trajectories, prep_trajectories)

        if parallel is None:
            mymap = map
        elif parallel == 'multiprocessing':
            mymap = Pool().map
        elif parallel == 'dtm':
            mymap = dtm.map
        else:
            raise ValueError('Unrecognized parallelization')


        # function that returns a list of random indices
        gen_sub_indices = lambda: np.array(random.sample(range(self.num_frames), self.num_frames / shrink_multiple))
        # gen_sub_indices = lambda: np.arange(self.num_frames)

        sub_indices = [gen_sub_indices() for i in range(num_samples)]
        ptrajs = [self.ptraj[sub_indices[i]] for i in range(num_samples)]

        clarans_args = uneven_zip(metric, ptrajs, k, num_local_minima, max_neighbors, local_swap, ['kcenters'], None, None, False)

        results = mymap(_clarans_helper, clarans_args)
        medoids_list, assignments_list, distances_list = zip(*results)
        best_i = np.argmin([np.sum(d) for d in distances_list])

        # print 'best i', best_i
        # print 'best medoids (relative to subindices)', medoids_list[best_i]
        # print 'sub indices', sub_indices[best_i]
        # print 'best_medoids', sub_indices[best_i][medoids_list[best_i]]
        self._generator_indices = sub_indices[best_i][medoids_list[best_i]]



class HybridKMedoids(BaseFlatClusterer):
    def __init__(self, metric, trajectories=None, prep_trajectories=None, k=None,
                 distance_cutoff=None, local_num_iters=10, global_num_iters=0,
                 norm_exponent=2.0, too_close_cutoff=.0001, ignore_max_objective=False):
        """Run the hybrid kmedoids clustering algorithm on a set of trajectories

        Parameters
        ----------
        metric : msmbuilder.metrics.AbstractDistanceMetric
            A metric capable of handling `ptraj`
        trajectory : Trajectory or list of msmbuilder.Trajectory
            data to cluster
        k : int
            number of desired clusters
        num_iters : int
            number of swaps to attempt per medoid
        local_swap : boolean, optional
            If true, proposed swaps will be between a medoid and a data point
            currently assigned to that medoid. If false, the data point for the
            proposed swap is selected randomly.
        norm_exponent : float, optional
            exponent to use in pnorm of the distance to generate objective function
        too_close_cutoff : float, optional
            Summarily reject proposed swaps if the distance of the medoid to the trial
            medoid is less than thus value
        ignore_max_objective : boolean, optional
            Ignore changes to the distance of the worst classified point, and only
            reject or accept swaps based on changes to the p norm of all the data
            points.

        References
        ----------
        .. [1] Beauchamp, K, et. al. MSMBuilder2

        See Also
        --------
        KCenters : faster, less accurate
        Clarans : slightly more clever termination criterion
        """

        super(HybridKMedoids, self).__init__(metric, trajectories, prep_trajectories)


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

# class KMeans(object):
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
