from __future__ import print_function, division, absolute_import
from mdtraj.utils.six import with_metaclass
import abc
import re
import numpy as np
import warnings

from scipy.spatial.distance import cdist, pdist


class AbstractDistanceMetric(with_metaclass(abc.ABCMeta, object)):

    """Abstract base class for distance metrics. All distance metrics should
    inherit from this abstract class.

    Provides a niave implementation of all_pairwise and one_to_many in terms
    of the abstract method one_to_all, which may be overridden by subclasses.
    """

    @abc.abstractmethod
    def prepare_trajectory(self, trajectory):
        """Prepare trajectory on a format that is more conventient to take
        distances on.

        Parameters
        ----------
        trajecory : msmbuilder.Trajectory
            Trajectory to prepare

        Returns
        -------
        prepared_traj : array-like
            the exact form of the prepared_traj is subclass specific, but it should
            support fancy indexing

        Notes
        -----
        For RMSD, this is going to mean making word-aligned padded
        arrays (TheoData) suitable for faste calculation, for dihedral-space
        distances means computing the dihedral angles, etc."""

        return

    @abc.abstractmethod
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

        return

    def one_to_many(self, prepared_traj1, prepared_traj2, index1, indices2):
        """Calculate the a vector of distances from the index1th frame of
        prepared_traj1 to all of the indices2 frames of prepared_traj2.

        Parameters
        ----------
        prepared_traj1 : prepared_trajectory
            First prepared trajectory
        prepared_traj2 : prepared_trajectory
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`
        indices2 : ndarray
            list of indices in `prepared_traj2` to calculate the distances to

        Returns
        -------
            Vector of distances of length len(indices2)

        Notes
        -----
        A subclass should be able to provide a more efficient implementation of
        this
        """

        return self.one_to_all(prepared_traj1, prepared_traj2[indices2], index1)

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

        traj_length = len(prepared_traj)
        output = -1 * np.ones(traj_length * (traj_length - 1) / 2)
        p = 0
        for i in range(traj_length):
            cmp_indices = np.arange(i + 1, traj_length)
            output[p: p + len(cmp_indices)] = self.one_to_many(prepared_traj,
                                                               prepared_traj, i, cmp_indices)
            p += len(cmp_indices)
        return output


class Vectorized(AbstractDistanceMetric):

    """Represent MSM frames as vectors in some arbitrary vector space, and then
    use standard vector space metrics. 

    Some examples of this might be extracting the contact map or dihedral angles.

    In order to be a full featured DistanceMetric, a subclass of
    Vectorized implements its own prepared_trajectory() method, Vectorized
    provides the remainder.

    allowable_scipy_metrics gives the list of metrics which your client
    can use. If the vector space that you're projecting your trajectory onto is 
    just a space of boolean vectors, then you probably don't want to allow eulcidean
    distance for instances.

    default_scipy_metric is the metric that will be used by your default metric
    if the user leaves the 'metric' field blank/unspecified.

    default_scipy_p is the default value of 'p' that will be used if left 
    unspecified. the value 'p' is ONLY used for the minkowski (pnorm) metric, so
    otherwise the scipy.spatial.distance code ignores it anyways.

    See http://docs.scipy.org/doc/scipy/reference/spatial.distance.html for a
    description of all the distance metrics and how they work.
    """

    allowable_scipy_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                               'correlation', 'cosine', 'euclidean', 'minkowski',
                               'sqeuclidean', 'dice', 'kulsinki', 'matching',
                               'rogerstanimoto', 'russellrao', 'sokalmichener',
                               'sokalsneath', 'yule', 'seuclidean', 'mahalanobis',
                               'sqmahalanobis']

    def __init__(self, metric='euclidean', p=2, V=None, VI=None):
        """Create a Vectorized metric

        Parameters
        ----------
        metric : {'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'euclidean', 'minkowski', 'sqeuclidean','dice', 'kulsinki', 'matching', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'yule', 'seuclidean', 'mahalanobis', 'sqmahalanobis'}
            Distance metric to equip the vector space with.
            See http://docs.scipy.org/doc/scipy/reference/spatial.distance.html
            for details
        p : int, optional
            p-norm order, used for metric='minkowski'
        V : ndarray, optional
            variances, used for metric='seuclidean'
        VI : ndarray, optional
            inverse covariance matrix, used for metric='mahalanobis'

        """

        self._validate_scipy_metric(metric)
        self.metric = metric
        self.p = p
        self.V = V
        self.VI = VI

        if self.metric == 'seuclidean' and V is None:
            raise ValueError('To use seuclidean, you need to supply V')
        if self.metric in ['mahalanobis', 'sqmahalanobis'] and VI is None:
            raise ValueError('To used mahalanobis or sqmahalanobis, you need to supply VI')

    def _validate_scipy_metric(self, metric):
        """Ensure that "metric" is an "allowable" metric (in allowable_scipy_metrics)"""
        if not metric in self.allowable_scipy_metrics:
            raise TypeError('%s is an  unrecognize metric. "metric" must be one of %s' %
                            (metric, str(self.allowable_scipy_metrics)))

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
        distances : ndarray
            Vector of distances of length len(indices2)
        """

        if not isinstance(index1, int):
            raise TypeError('index1 must be of type int.')
        out = cdist(prepared_traj2[indices2], prepared_traj1[[index1]],
                    metric=self.metric, p=self.p, V=self.V, VI=self.VI)

        return out[:, 0]

    def one_to_all(self, prepared_traj1, prepared_traj2, index1):
        """Measure the distance from one frame to every frame in a trajectory

        The distances calculated are from the `index1`th frame of `prepared_traj1`
        to all the frames in `prepared_traj2` with indices `indices2`. Although
        this is similar to one_to_many, it can often be computed faster

        Parameters
        ----------
        prepared_traj1 : ndarray
            First prepared trajectory
        prepared_traj2 : ndarray
            Second prepared trajectory
        index1 : int
            index in `prepared_trajectory`


        Returns
        -------
        distances : ndarray
            A vector of distances of length len(prepared_traj2)"""

        if not isinstance(index1, int):
            raise TypeError('index1 must be of type int.')
        out2 = cdist(prepared_traj2, prepared_traj1[[index1]], metric=self.metric,
                     p=self.p, V=self.V, VI=self.VI)
        return out2[:, 0]

    def many_to_many(self, prepared_traj1, prepared_traj2, indices1, indices2):
        """Get a matrix of distances from each frame in a set to each other frame
        in a second set.

        Calculate a MATRIX of distances from the frames in prepared_traj1 with
        indices `indices1` to the frames in prepared_traj2 with indices `indices2`,
        using supplied metric.

        Parameters
        ----------
        prepared_traj1 : ndarray
            First prepared trajectory
        prepared_traj2 : ndarray
            Second prepared trajectory
        indices1 : array_like
            list of indices in `prepared_traj1` to calculate the distances from
        indices2 : array_like
            list of indices in `prepared_traj2` to calculate the distances to

        Returns
        -------
        distances : ndarray
            A 2D array of shape len(indices1) * len(indices2)"""

        out = cdist(prepared_traj1[indices1], prepared_traj2[indices2], metric=self.metric,
                    p=self.p, V=self.V, VI=self.VI)
        return out

    def all_to_all(self, prepared_traj1, prepared_traj2):
        """Get a matrix of distances from all frames in one traj to all frames in
        another


        Parameters
        ----------
        prepared_traj1 : ndarray
            First prepared trajectory
        prepared_traj2 : ndarray
            Second prepared trajectory

        Returns
        -------
        distances : ndarray
            A 2D array of shape len(preprared_traj1) * len(preprared_traj2)"""

        if prepared_traj1 is prepared_traj2:
            warnings.warn('runtime', re.sub("\s+", " ", """it's not recommended to
            use this method to calculate the full pairwise distance matrix for
            one trajectory to itself (as you're doing). Use all_pairwise, which
            will be more efficient if you reall need the results as a 2D matrix
            (why?) then you can always use scipy.spatial.distance.squareform()
            on the output of all_pairwise()""".replace('\n', ' ')))

        out = cdist(prepared_traj1, prepared_traj2, metric=self.metric, p=self.p,
                    V=self.V, VI=self.VI)
        return out

    def all_pairwise(self, prepared_traj):
        """Calculate a condense" distance matrix of all the pairwise distances
        between each frame with each other frame in prepared_traj

        The condensed distance matrix can be converted to the redundant square form
        if desired

        Parameters
        ----------
        prepared_traj1 : ndarray
            Prepared trajectory

        Returns
        -------
        distances : ndarray
            1D vector of length len(pairwise_traj) choose 2 where the i*jth
            entry contains the distance between prepared_traj[i] and prepared_traj[j]

        See Also
        --------
        scipy.spatial.distance.pdist
        scipy.spatial.distance.squareform
        """

        out = pdist(prepared_traj, metric=self.metric, p=self.p,
                    V=self.V, VI=self.VI)
        return out
