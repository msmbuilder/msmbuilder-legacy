import numpy as np
import logging
logger = logging.getLogger(__name__)
from baseclasses import AbstractDistanceMetric


class Hybrid(AbstractDistanceMetric):
    "A linear combination of other distance metrics"

    class HybridPreparedTrajectory(object):
        """Container to to hold the prepared trajectory.
        This container needs to support slice notation in a way that kind
        of passes through the indices to the 2nd dimension. So if you
        have a HybridPreparedTrajectory with 3 bases metrics, and you
        do metric[0:100], it needs to return a HybridPrepareTrajectory
        with the same three base metrics, but with each of the base prepared
        trajectories sliced 0:100. We don't want to slice out base_metrics
        and thus have metric[0] return only one of the three prepared_trajectories
        in its full length."""
        def __init__(self, *args):
            self.num_base = len(args)
            self.length = len(args[0])
            if not np.all((len(arg) == self.length for arg in args)):
                raise ValueError("Must all be equal length")

            self.datas = args


        def __getitem__(self, key):
            if isinstance(key, int):
                key = slice(key, key + 1)
            return Hybrid.HybridPreparedTrajectory(*(d[key] for d in self.datas))

        def __len__(self):
            return self.length

        def __setitem__(self, key, value):
            try:
                if self.num_base != value.num_base:
                    raise ValueError("Must be prepared over the same metrics")
            except:
                raise ValueError("I can only set in something which is also a HybridPreparedTrajectory")

            for i in xrange(self.num_base):
                self.datas[i][key] = value.datas[i]



    def __init__(self, base_metrics, weights):
        """Create a hybrid linear combinatiin distance metric

        Parameters
        ----------
        base_metrics : list of distance metric objects
        weights : list of floats
            list of scalars of equal length to `base_metrics` -- each base
            metric will be multiplied by that scalar when they get summed.
        """

        self.base_metrics = base_metrics
        self.weights = weights
        self.num = len(self.base_metrics)

        if not len(self.weights) == self.num:
            raise ValueError()


    def prepare_trajectory(self, trajectory):
        """Preprocess trajectory for use with this metric

        Parameters
        ----------
        trajectory : mdtraj.Trajectory
            Trajectory to prepare

        Returns
        -------
        prepared_trajectory : array_like
            The prepared trajectory is a special array like object called
            HybridPreparedTrajectory which is designed to pass through the slicing
            correctly so that if you ask for prepared_trajectory[5] you get the
            appropriate 5th frames dihedral angles, RMSD, etc (depending what
            base metrics you used)
        """
        prepared = (m.prepare_trajectory(trajectory) for m in self.base_metrics)
        return self.HybridPreparedTrajectory(*prepared)


    def one_to_many(self, prepared_traj1, prepared_traj2, index1, indices2):
        """Calculate a vector of distances from one frame of the first trajectory
        to many frames of the second trajectory

        The distances calculated are from the `index1`th frame of `prepared_traj1`
        to the frames in `prepared_traj2` with indices `indices2`

        Parameters
        ----------
        prepared_traj1 : ndarray
            First prepared trajectory
        prepared_traj2 : ndarray
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`
        indices2 : ndarray
            list of indices in `prepared_traj2` to calculate the distances to

        Returns
        -------
        Vector of distances of length len(indices2)
        """
        distances = None
        for i in range(self.num):
            d = self.base_metrics[i].one_to_many(prepared_traj1.datas[i], prepared_traj2.datas[i], index1, indices2)
            if distances is None:
                distances = self.weights[i] * d
            else:
                distances += self.weights[i] * d
        return distances


    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        """Calculate the vector of distances from the index1th frame of
        prepared_traj1 to all of the frames in prepared_traj2.

        Parameters
        ----------
        prepared_traj1 : prepared_trajectory
            First prepared trajectory
        prepared_traj2 : prepared_trajectory
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`

        Returns
        -------
        distances : ndarray
            vector of distances of length len(prepared_traj2)

        Notes
        -----
        Although this might seem to be a special case of one_to_many(), it
        can often be implemented in a much more optimized way because it doesn't
        require construction of the indices2 array and array slicing in python
        is kindof slow.
        """

        distances = None
        for i in range(self.num):
            d = self.base_metrics[i].one_to_all(prepared_traj1.datas[i], prepared_traj2.datas[i], index1)
            if distances is None:
                distances = self.weights[i] * d
            else:
                distances += self.weights[i] * d
        return distances


    def all_pairwise(self, prepared_traj):
        """Calculate condensed distance metric of all pairwise distances

        See `scipy.spatial.distance.squareform` for information on how to convert
        the condensed distance matrix to a redundant square matrix

        Parameters
        ----------
        prepared_traj : array_like
            Prepared trajectory

        Returns
        -------
        Y : ndarray
            A 1D array containing the distance from each frame to each other frame

        See Also
        --------
        fast_pdist
        scipy.spatial.distance.squareform
        """

        distances = None
        for i in range(self.num):
            d = self.base_metrics[i].all_pairwise(prepared_traj.datas[i])
            distances = self.weights[i] * d if distances is None else distances + self.weights[i] * d
        return distances


class HybridPNorm(Hybrid):
    """A p-norm combination of other distance metrics. With p=2 for instance,
    this gives you the root mean square combination of the base metrics"""

    def __init__(self, base_metrics, weights, p=2):
        """Initialize the HybridPNorm distance metric.

        Parameters
        ----------
        base_metrics : list of distance metric objects
        weights : list of floats
            list of scalars of equal length to `base_metrics` -- each base
            metric will be multiplied by that scalar.
        p : float
            p should be a scalar, greater than 0, which will be the exponent.
            If p=2, all the base metrics will be squared, then summed, then the
            square root will be taken. If p=3, the base metrics will be cubed,
            summed and cube rooted, etc.
        """

        self.p = float(p)
        super(HybridPNorm, self).__init__(base_metrics, weights)


    def one_to_many(self, prepared_traj1, prepared_traj2, index1, indices2):
        """Calculate a vector of distances from one frame of the first trajectory
        to many frames of the second trajectory

        The distances calculated are from the `index1`th frame of `prepared_traj1`
        to the frames in `prepared_traj2` with indices `indices2`

        Parameters
        ----------
        prepared_traj1 : ndarray
            First prepared trajectory
        prepared_traj2 : ndarray
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`
        indices2 : ndarray
            list of indices in `prepared_traj2` to calculate the distances to

        Returns
        -------
        Vector of distances of length len(indices2)
        """

        distances = None
        for i in range(self.num):
            d = self.base_metrics[i].one_to_many(prepared_traj1.datas[i], prepared_traj2.datas[i], index1, indices2)
            if distances is None:
                distances = (self.weights[i] * d) ** self.p
            else:
                distances += (self.weights[i] * d) ** self.p
        return distances ** (1.0 / self.p)

    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        """Calculate the vector of distances from the index1th frame of
        prepared_traj1 to all of the frames in prepared_traj2.

        Parameters
        ----------
        prepared_traj1 : prepared_trajectory
            First prepared trajectory
        prepared_traj2 : prepared_trajectory
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`

        Returns
        -------
        distances : ndarray
            vector of distances of length len(prepared_traj2)

        Notes
        -----
        Although this might seem to be a special case of one_to_many(), it
        can often be implemented in a much more optimized way because it doesn't
        require construction of the indices2 array and array slicing in python
        is kindof slow.
        """

        distances = None
        for i in range(self.num):
            d = self.base_metrics[i].one_to_all(prepared_traj1.datas[i], prepared_traj2.datas[i], index1)
            if distances is None:
                distances = (self.weights[i] * d) ** self.p
            else:
                distances += (self.weights[i] * d) ** self.p
        return distances ** (1.0 / self.p)

    def all_pairwise(self, prepared_traj):
        """Calculate condensed distance metric of all pairwise distances

        See `scipy.spatial.distance.squareform` for information on how to convert
        the condensed distance matrix to a redundant square matrix

        Parameters
        ----------
        prepared_traj : array_like
            Prepared trajectory

        Returns
        -------
        Y : ndarray
            A 1D array containing the distance from each frame to each other frame

        See Also
        --------
        fast_pdist
        scipy.spatial.distance.squareform
        """

        distances = None
        for i in range(self.num):
            d = self.base_metrics[i].all_pairwise(prepared_traj.datas[i])
            d = (self.weights[i] * d) ** self.p
            distances = d if distances is None else distances + (self.weights[i] * d)
            logger.info('got %s', i)
        return distances ** (1.0 / self.p)
