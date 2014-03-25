"""
Compute the drift in trajectories under different distance metrics
using new distance_metrics.py code
"""
from __future__ import print_function, division, absolute_import
import numpy as np


def _drift_single_trajectory(metric, trajectory, tau):
    """
    Compute the drift in your desired metric between all pairs of
    conformations in the supplied trajectory which are seperated
    by tau frames.

    Parameters
    ----------
    metric : msmbuilder.metrics.AbstractDistanceMetric
        The distance metric to compute distances with
    trajectory : msmbuilder.Trajectory
        The trajectory to compute the distances
    tau : {int, np.ndarray}
        tau can be either a positive integer or an array of positive
        integers. If tau is an integer, the return value is a 1D array
        of length equal to the length of the trajectory minus tau
        containing the pairwise distance of all conformations in
        trajectory seperated by tau frames. If tau is an array of
        integers of length n, the return value is a 2D array with n
        rows and a number of columns equal to the length of the
        trajectory minus min(tau). The ith row of the returned array
        contains the pairwise distance of all of the conformations in
        the supplied trajectory separated by tau[i] frames. The final
        i entries in row i will be padded with -1s to ensure that the
        output 2D arrau is rectangular.

    Returns
    -------
    distances : np.ndarray
        1D or 2D array of the drifts, depending on whether tau is a single
        number or an array
    """
    # make sure tau is a 1D numpy array of positive ints, or make it into one
    tau = __typecheck_tau(tau)
    # if not isinstance(metric, AbstractDistanceMetric):
    #   raise TypeError('metric must be an instance of AbstractDistanceMetric. you supplied a %s' % metric)

    traj_length = trajectory['XYZList'].shape[0]
    ptraj = metric.prepare_trajectory(trajectory)
    distances = -1 * np.ones((len(tau), traj_length - np.min(tau)))

    for i in xrange(traj_length - np.min(tau)):
        comp_indices = [elem for elem in tau + i if elem < traj_length]
        d = metric.one_to_many(ptraj, ptraj, i, comp_indices)
        # these distances are the ith column
        distances[0:len(comp_indices), i] = d

    # if there was only 1 element in tau, reshape output so its 1D
    # if distances.shape == (1, traj_length - np.min(tau)):
    #    distances = np.reshape(distances, traj_length - np.min(tau))
    return distances


def drift(metric, trajectories, taus):
    if 'XYZList' in trajectories:
        trajectories = [trajectories]
    if isinstance(taus, int):
        taus = [taus]

    output = [np.array([])] * len(taus)
    for trajectory in trajectories:
        d = _drift_single_trajectory(metric, trajectory, taus)
        for i in range(len(taus)):
            #length_i = d.shape[1] - (taus[i] - np.min(taus))
            selected_d = np.ma.masked_less(d[i, :], 0)
            selected_d = np.ma.compressed(selected_d)
            output[i] = np.hstack((output[i], selected_d))

    return output


def square_drift(metric, trajectories, tau):
    output = drift(metric, trajectories, tau)
    for i, row in enumerate(output):
        output[i] = row ** 2
    return output


def get_epsilon_neighborhoods(metric, ptraj, tau):

    output = []
    N = len(ptraj)
    for i in xrange(N):
        if i < tau:
            output.append(metric.one_to_many(ptraj, ptraj,
                                             i, [i + tau])[0])
        elif i >= N - tau:
            output.append(metric.one_to_many(ptraj, ptraj, i,
                                             [i - tau])[0])
        else:
            output.append(metric.one_to_many(ptraj, ptraj, i,
                                             [i - tau, i + tau]).max())

    return output


def hitting_time(metric, trajectory, epsilon):
    """
    For each frame in trajectories, determine the time it takes to go more
    than epsilon distance from where it started.

    Returns: a masked array of integers of length equal to the length of
    the trajectory. The masked values correspond to frames for which the
    hitting time could not be determined (maybe the trajectory wasn't long
    enough so it never left the epsilon-ball)
    """

    traj_length = len(trajectory)
    ptraj = metric.prepare_trajectory(trajectory)
    window_length = 8
    output = -1 * np.ones(len(trajectory), dtype=np.int)

    for i in xrange(traj_length):
        found = False
        mult = 1
        start = i + 1
        while not found:
            forward_window = np.arange(start, start + window_length * mult)
            forward_window = forward_window[np.where(forward_window < traj_length)]
            if len(forward_window) == 0:
                break
            # print forward_window
            d = metric.one_to_many(ptraj, ptraj, i, forward_window)
            where = np.where(d > epsilon)[0]
            found = len(where) != 0
            if not found:
                start += window_length * mult
                mult *= 2
        if found:
            first = np.min(where)
            output[i] = forward_window[first] - i
            # print d[first - 1], d[first]
            # xprint i, first, forward_window
            # print i
            # print forward_window[np.min(where)]
            # print d[np.min(where)]

    # return np.ma.masked_equal(output, -1)
    return output


def __typecheck_tau(tau):
    """make sure tau is a 1D numpy array of positive ints, 
    or make it into one if possible unambiguously"""

    if isinstance(tau, int):
        if tau < 0:
            raise TypeError('Tau cannot be negative')

        tau = np.array([tau])
    else:
        tau = np.array(tau)
        if not len(tau.shape) == 1:
            raise TypeError('Tau must be a 1D array or an int. You supplied %s' % tau)

    # ensure positive
    if not np.all(tau == np.abs(tau)):
        raise TypeError('Taus must be all positive.')

    # ensure ints
    if not np.all(tau == np.array(tau, dtype='int')):
        raise TypeError('Taus must be all integers.')

    return tau
